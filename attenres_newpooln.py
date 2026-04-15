import os
import os.path as osp
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import argparse
from torch.cuda.amp import autocast, GradScaler

NUM_CLASSES = 6
TARGET_LEN = 384

class RFISet2D(Dataset):
    def __init__(self, root2d, split):
        self.img_dir = osp.join(root2d, f"{split}_npz")
        self.files = [f for f in os.listdir(self.img_dir) if f.endswith(".npz")] if osp.isdir(self.img_dir) else []
        if len(self.files) == 0:
            print(f"警告: 数据集目录 {self.img_dir} 为空或不存在，或者没有 .npz 文件！")
        
    def __len__(self):
        return len(self.files)
        
    def __getitem__(self, idx):
        f = self.files[idx]
        p2d = osp.join(self.img_dir, f)
        with np.load(p2d) as z:
            if 'image' in z: im = z['image']
            elif 'arr_0' in z: im = z['arr_0']
            else: im = list(z.values())[0]
            lab = z['label'] if 'label' in z else None
            
        im = im.astype(np.float32)
        lab = lab.astype(np.int64)
        
        im = np.log1p(im - np.min(im) if np.min(im) < 0 else im)
        im = (im - np.mean(im)) / (np.std(im) + 1e-8)
        
        if im.ndim == 3:
            im = im.sum(axis=2)
        if lab.ndim != 2:
            lab = lab.squeeze()
            
        h, w = im.shape
        if w != TARGET_LEN:
            t = torch.tensor(im).unsqueeze(0).unsqueeze(0)
            t = F.interpolate(t, size=(h, TARGET_LEN), mode="bilinear", align_corners=False)
            im = t.squeeze(0).squeeze(0).numpy()
            
        x = torch.tensor(im, dtype=torch.float32).unsqueeze(0)
        y = torch.tensor(lab, dtype=torch.int64)
        return x, y, f

class StripSEBlock2D(nn.Module):
    def __init__(self, c, r=16):
        super().__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        mip = max(c // r, 1)
        self.conv = nn.Conv2d(c, mip, 1)
        self.act = nn.ReLU(inplace=True)
        self.conv_h = nn.Conv2d(mip, c, 1)
        self.conv_w = nn.Conv2d(mip, c, 1)
        self.gate = nn.Sigmoid()
        
    def forward(self, x):
        b, c, h, w = x.shape
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)
        
        y = torch.cat([x_h, x_w], dim=2)
        y = self.act(self.conv(y))
        
        y_h, y_w = torch.split(y, [h, w], dim=2)
        y_w = y_w.permute(0, 1, 3, 2)
        
        a_h = self.gate(self.conv_h(y_h))
        a_w = self.gate(self.conv_w(y_w))
        return x * a_h * a_w

class BlockSummary2D(nn.Module):
    def __init__(self, out_dim=256, in_channels_list=None, strip_len=8):
        super().__init__()
        in_channels_list = in_channels_list or []
        self.strip_len = strip_len
        self.proj = nn.ModuleDict({
            str(c): nn.Linear(c * 2 * strip_len, out_dim, bias=False) 
            for c in sorted(set(in_channels_list))
        })
        
    def forward(self, x):
        b, c, h, w = x.shape
        s_h = F.adaptive_avg_pool2d(x, (self.strip_len, 1)).view(b, c * self.strip_len)
        s_w = F.adaptive_avg_pool2d(x, (1, self.strip_len)).view(b, c * self.strip_len)
        
        strip_feat = torch.cat([s_h, s_w], dim=1)
        summary = self.proj[str(c)](strip_feat)
        return summary.view(b, -1, 1, 1)

class DepthAttn2D(nn.Module):
    def __init__(self, out_c, summary_dim=256, num_heads=4, block_size=8, strip_len=8):
        super().__init__()
        self.summary_dim = summary_dim
        self.num_heads = num_heads
        self.head_dim = summary_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.tau = nn.Parameter(torch.tensor(1.0))
        self.block_size = block_size
        self.strip_len = strip_len
        
        self.q_norm = nn.RMSNorm(out_c * 2 * strip_len)
        self.q_proj = nn.Linear(out_c * 2 * strip_len, summary_dim)
        
        self.k_norm = nn.RMSNorm(summary_dim)
        self.v_norm = nn.RMSNorm(summary_dim)
        self.k_proj = nn.Linear(summary_dim, summary_dim)
        self.v_proj = nn.Linear(summary_dim, summary_dim)
        self.out_proj = nn.Linear(summary_dim, out_c)
        
    def group_blocks(self, summaries):
        if self.block_size is None or self.block_size <= 0:
            return summaries
        blocks = []
        for i in range(0, len(summaries), self.block_size):
            chunk = summaries[i:i + self.block_size]
            if len(chunk) == 1:
                blocks.append(chunk[0])
            else:
                blocks.append(torch.mean(torch.stack(chunk, dim=0), dim=0))
        return blocks
        
    def forward(self, summaries, y):
        if summaries is None or len(summaries) == 0:
            return None
        b, c, h, w = y.shape
        summaries = self.group_blocks(summaries)
        
        s_h = F.adaptive_avg_pool2d(y, (self.strip_len, 1)).view(b, c * self.strip_len)
        s_w = F.adaptive_avg_pool2d(y, (1, self.strip_len)).view(b, c * self.strip_len)
        q_feat = torch.cat([s_h, s_w], dim=1)
        
        q = self.q_norm(q_feat)
        q = self.q_proj(q)
        
        k_list = []
        v_list = []
        for s in summaries:
            sv = s.flatten(1)
            k_list.append(self.k_proj(self.k_norm(sv)))
            v_list.append(self.v_proj(self.v_norm(sv)))
        k = torch.stack(k_list, dim=1)
        v = torch.stack(v_list, dim=1)
        q = q.view(b, self.num_heads, 1, self.head_dim)
        k = k.view(b, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.view(b, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        scores = torch.matmul(q, k.transpose(-2, -1)).squeeze(-2)
        scores = scores * (self.scale * self.tau)
        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn.unsqueeze(-2), v).squeeze(-2)
        out = out.permute(0, 2, 1).contiguous().view(b, self.summary_dim)
        out = self.out_proj(out).view(b, c, 1, 1)
        out = F.interpolate(out, size=(h, w), mode="bilinear", align_corners=False)
        return out

class GateFusion2D(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.fc1 = nn.Conv2d(c * 2, c, 1)
        self.act = nn.ReLU(inplace=True)
        self.fc2_h = nn.Conv2d(c, c, 1)
        self.fc2_w = nn.Conv2d(c, c, 1)
        self.gate = nn.Sigmoid()
        
    def forward(self, y, attn):
        if attn is None:
            return y
        b, c, h, w = y.shape
        cat_feat = torch.cat([y, attn], dim=1)
        
        pool_h = self.pool_h(cat_feat)
        pool_w = self.pool_w(cat_feat).permute(0, 1, 3, 2)
        
        pool_cat = torch.cat([pool_h, pool_w], dim=2)
        mid = self.act(self.fc1(pool_cat))
        
        mid_h, mid_w = torch.split(mid, [h, w], dim=2)
        mid_w = mid_w.permute(0, 1, 3, 2)
        
        g_h = self.gate(self.fc2_h(mid_h))
        g_w = self.gate(self.fc2_w(mid_w))
        g = g_h * g_w
        
        return y * (1 - g) + attn * g

class AttenResBlock(nn.Module):
    def __init__(self, in_c, out_c, in_channels_list, summary_dim=256, num_heads=4, block_size=8, strip_len=8):
        super().__init__()
        self.c1 = nn.Conv2d(in_c, out_c, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.c2 = nn.Conv2d(out_c, out_c, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.act = nn.ReLU(inplace=True)
        self.se = StripSEBlock2D(out_c)
        self.depth_attn = DepthAttn2D(out_c, summary_dim=summary_dim, num_heads=num_heads, block_size=block_size, strip_len=strip_len)
        self.gate_fuse = GateFusion2D(out_c)
        
    def forward(self, x, prev_summaries=None):
        y = self.act(self.bn1(self.c1(x)))
        y = self.bn2(self.c2(y))
        y = self.se(y)
        attn = None
        if prev_summaries:
            attn = self.depth_attn(prev_summaries, y)
        y = self.gate_fuse(y, attn)
        y = self.act(y)
        return y

class UpBlock(nn.Module):
    def __init__(self, in_c, out_c, in_channels_list, strip_len=8, block_size=8):
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, bias=False)
        )
        self.bn = nn.BatchNorm2d(out_c)
        self.act = nn.ReLU(inplace=True)
        self.block = AttenResBlock(out_c * 2, out_c, in_channels_list, strip_len=strip_len, block_size=block_size)
        
    def forward(self, x, skip, prev_summaries=None):
        y = self.act(self.bn(self.up(x)))
        if (y.shape[-2] != skip.shape[-2]) or (y.shape[-1] != skip.shape[-1]):
            y = F.interpolate(y, size=(skip.shape[-2], skip.shape[-1]), mode="bilinear", align_corners=False)
        y = torch.cat([y, skip], dim=1)
        y = self.block(y, prev_summaries)
        return y

class AttenResUNet(nn.Module):
    def __init__(self, in_c=1, base=64, strip_len=8, block_size=8):
        super().__init__()
        channels = [base, base * 2, base * 4, base * 8, base * 16]
        self.summary = BlockSummary2D(out_dim=256, in_channels_list=channels, strip_len=strip_len)
        self.e1 = AttenResBlock(in_c, base, channels, strip_len=strip_len, block_size=block_size)
        self.p1 = nn.MaxPool2d(2)
        self.e2 = AttenResBlock(base, base * 2, channels, strip_len=strip_len, block_size=block_size)
        self.p2 = nn.MaxPool2d(2)
        self.e3 = AttenResBlock(base * 2, base * 4, channels, strip_len=strip_len, block_size=block_size)
        self.p3 = nn.MaxPool2d(2)
        self.e4 = AttenResBlock(base * 4, base * 8, channels, strip_len=strip_len, block_size=block_size)
        self.p4 = nn.MaxPool2d(2)
        self.b = AttenResBlock(base * 8, base * 16, channels, strip_len=strip_len, block_size=block_size)
        self.u4 = UpBlock(base * 16, base * 8, channels, strip_len=strip_len, block_size=block_size)
        self.u3 = UpBlock(base * 8, base * 4, channels, strip_len=strip_len, block_size=block_size)
        self.u2 = UpBlock(base * 4, base * 2, channels, strip_len=strip_len, block_size=block_size)
        self.u1 = UpBlock(base * 2, base, channels, strip_len=strip_len, block_size=block_size)
        self.out = nn.Conv2d(base, NUM_CLASSES, 1)
        
    def forward(self, x):
        prev = []
        s1 = self.e1(x, prev); prev.append(self.summary(s1)); p1 = self.p1(s1)
        s2 = self.e2(p1, prev); prev.append(self.summary(s2)); p2 = self.p2(s2)
        s3 = self.e3(p2, prev); prev.append(self.summary(s3)); p3 = self.p3(s3)
        s4 = self.e4(p3, prev); prev.append(self.summary(s4)); p4 = self.p4(s4)
        b = self.b(p4, prev); prev.append(self.summary(b))
        y = self.u4(b, s4, prev); prev.append(self.summary(y))
        y = self.u3(y, s3, prev); prev.append(self.summary(y))
        y = self.u2(y, s2, prev); prev.append(self.summary(y))
        y = self.u1(y, s1, prev); prev.append(self.summary(y))
        return self.out(y)

def one_hot(y, c):
    return F.one_hot(y.long(), c).permute(0,3,1,2).float()

def dice_loss(logits, y, c):
    p = torch.softmax(logits, dim=1)
    t = one_hot(y, c).to(p.device)
    num = (2 * (p * t).sum(dim=(0,2,3)) + 1e-6)
    den = (p.sum(dim=(0,2,3)) + t.sum(dim=(0,2,3)) + 1e-6)
    d = 1 - (num / den)
    return d.mean()

def focal_loss(logits, y, w_cls, gamma=2.0):
    ce_loss = F.cross_entropy(logits, y, weight=w_cls, reduction='none')
    pt = torch.exp(-ce_loss)
    f_loss = ((1 - pt) ** gamma) * ce_loss
    return f_loss.mean()

def compute_class_priors(ds):
    counts = torch.zeros(NUM_CLASSES, dtype=torch.float64); total = 0
    for i in range(len(ds)):
        _, y2d, _ = ds[i]
        y_flat = y2d.to(torch.int64).flatten()
        counts += torch.bincount(y_flat, minlength=NUM_CLASSES).double()
        total += y_flat.numel()
    priors = (counts / max(total, 1)).float()
    return priors, counts.float()

def val_metrics(model, loader, device, w_cls, desc):
    model.eval()
    dice_num = torch.zeros(NUM_CLASSES, device=device)
    dice_den = torch.zeros(NUM_CLASSES, device=device)
    ce_sum = 0.0; n = 0
    class_total = torch.zeros(NUM_CLASSES, device=device)
    class_correct = torch.zeros(NUM_CLASSES, device=device)
    overall_total = 0; overall_correct = 0
    tp = torch.zeros(NUM_CLASSES, device=device)
    fp = torch.zeros(NUM_CLASSES, device=device)
    fn = torch.zeros(NUM_CLASSES, device=device)
    with torch.no_grad():
        for x2d, y2d, _ in tqdm(loader, desc=desc):
            x2d = x2d.to(device); y2d = y2d.to(device)
            with autocast():
                logits = model(x2d)
                ce_sum += F.cross_entropy(logits, y2d, weight=w_cls).item()
                
            p = torch.softmax(logits, dim=1)
            t = one_hot(y2d, NUM_CLASSES).to(device)
            dice_num += (2 * (p * t).sum(dim=(0,2,3)))
            dice_den += (p.sum(dim=(0,2,3)) + t.sum(dim=(0,2,3)))
            pred = p.argmax(dim=1)
            overall_correct += (pred == y2d).sum().item()
            overall_total += y2d.numel()
            
            for c in range(NUM_CLASSES):
                mask = (y2d == c)
                class_total[c] += mask.sum()
                class_correct[c] += ((pred == c) & mask).sum()
                tp[c] += ((pred == c) & (y2d == c)).sum()
                fp[c] += ((pred == c) & (y2d != c)).sum()
                fn[c] += ((pred != c) & (y2d == c)).sum()
            n += 1
            
    dices = (dice_num + 1e-6) / (dice_den + 1e-6)
    class_acc = torch.where(class_total > 0, class_correct / class_total, torch.zeros_like(class_total))
    overall_acc = overall_correct / max(overall_total, 1)
    prec = tp / torch.clamp(tp + fp, min=1)
    rec = tp / torch.clamp(tp + fn, min=1)
    f1 = (2 * prec * rec) / torch.clamp(prec + rec, min=1e-6)
    return dices.detach().cpu().numpy(), (ce_sum / max(n,1)), class_acc.detach().cpu().numpy(), overall_acc, prec.detach().cpu().numpy(), rec.detach().cpu().numpy(), f1.detach().cpu().numpy()

def train(batch_size, lr, epochs, root2d, save_dir, num_workers, block_size=8, base_c=32):
    torch.backends.cudnn.benchmark = True
    
    ds_tr = RFISet2D(root2d, "train")
    ds_va = RFISet2D(root2d, "val")
    ds_te = RFISet2D(root2d, "test")
    dl_tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    dl_va = DataLoader(ds_va, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    dl_te = DataLoader(ds_te, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AttenResUNet(1, base_c, block_size=block_size).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    
    warmup_epochs = 5
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs - warmup_epochs)
    
    priors, _ = compute_class_priors(ds_tr)
    w_cb = (torch.clamp(priors, min=1e-6) ** (-0.5)); w_cb = (w_cb / w_cb.mean()).to(device)
    os.makedirs(save_dir, exist_ok=True)
    
    scaler = GradScaler()
    best_score = -1.0; best_path = None
    for ep in range(1, epochs+1):
        model.train()
        pbar = tqdm(dl_tr, desc=f"训练 Epoch {ep}/{epochs}", total=len(dl_tr))
        
        if ep <= warmup_epochs:
            lr_scale = ep / warmup_epochs
            for pg in opt.param_groups:
                pg['lr'] = lr * lr_scale
                
        for x2d, y2d, _ in pbar:
            x2d = x2d.to(device); y2d = y2d.to(device)
            
            with autocast():
                logits = model(x2d)
                fl = focal_loss(logits, y2d, w_cb)
                d = dice_loss(logits, y2d, NUM_CLASSES)
                loss = fl + d
                
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(opt)
            scaler.update()
            
            pbar.set_postfix(loss="{:.4f}".format(loss.item()))
            
        if ep > warmup_epochs:
            scheduler.step()
            
        val_dices, val_ce, val_class_acc, val_overall_acc, val_class_prec, val_class_rec, val_class_f1 = val_metrics(model, dl_va, device, w_cb, "验证集评估")
        
        msg = "epoch {} | ".format(ep) \
              + " ".join(["val_prec_c{}:{:.4f}".format(i, val_class_prec[i]) for i in range(NUM_CLASSES)]) \
              + " " \
              + " ".join(["val_rec_c{}:{:.4f}".format(i, val_class_rec[i]) for i in range(NUM_CLASSES)]) \
              + " " \
              + " ".join(["val_f1_c{}:{:.4f}".format(i, val_class_f1[i]) for i in range(NUM_CLASSES)])
        print(msg)
        with open(osp.join(save_dir, "train_log.txt"), "a", encoding="utf-8") as f:
            f.write(msg + "\n")
            
        val_f1_mean = float(np.mean(val_class_f1[1:]))
        
        if val_f1_mean > best_score:
            best_score = val_f1_mean
            best_path = osp.join(save_dir, f"attenres_newpool_best_epoch{ep}.pt")
            torch.save(model.state_dict(), best_path)
            
            test_dices, test_ce, test_class_acc, test_overall_acc, test_class_prec, test_class_rec, test_class_f1 = val_metrics(model, dl_te, device, w_cb, "测试集评估")
            test_msg = "best updated -> epoch {} | val_f1_mean {:.4f} | ".format(ep, val_f1_mean) \
                       + " ".join(["test_prec_c{}:{:.4f}".format(i, test_class_prec[i]) for i in range(NUM_CLASSES)]) \
                       + " " \
                       + " ".join(["test_rec_c{}:{:.4f}".format(i, test_class_rec[i]) for i in range(NUM_CLASSES)]) \
                       + " " \
                       + " ".join(["test_f1_c{}:{:.4f}".format(i, test_class_f1[i]) for i in range(NUM_CLASSES)])
            print(test_msg)
            with open(osp.join(save_dir, "train_log.txt"), "a", encoding="utf-8") as f:
                f.write(test_msg + "\n")
            
        if ep % 10 == 0:
            torch.save(model.state_dict(), osp.join(save_dir, f"attenres_newpool_epoch{ep}.pt"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root2d", type=str, default=".\\RFI384")
    parser.add_argument("--save-dir", type=str, default=".\\checkpoints_attenres_newpooln")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--block-size", type=int, default=8, help="Block size for DepthAttn2D group_blocks")
    parser.add_argument("--base-c", type=int, default=32, help="Base channels")
    args = parser.parse_args()
    train(args.batch_size, args.lr, args.epochs, args.root2d, args.save_dir, args.num_workers, args.block_size, args.base_c)
