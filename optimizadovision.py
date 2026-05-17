"""
TT prueba optimizada para vision
"""

import math, struct, array, random, os, sys, time, json
import zipfile, tarfile, urllib.request, urllib.error
import numpy as np

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


# Mantenemos el nombre para compatibilidad

def Tensor(shape, data=None, fill=0.0):
    #Crea un ndarray float32 con la forma dada
    if data is not None:
        arr = np.array(data, dtype=np.float32).reshape(shape)
    else:
        arr = np.full(shape, fill, dtype=np.float32)
    return arr


# Funciones matemáticas 
def kaiming_init(fan_in, size):
    std = math.sqrt(2.0 / fan_in)
    return np.random.normal(0.0, std, size=size).astype(np.float32)

def softmax_2d(logits):
    m = np.max(logits, axis=0, keepdims=True) if logits.ndim > 1 else np.max(logits)
    e = np.exp(logits - m)
    return e / (np.sum(e, axis=0, keepdims=True) if logits.ndim > 1 else e.sum())



# Conv2d  im2col vectorizado

def _im2col(x_pad, K, S, oH, oW):
    #Convierte imagen paddeada a matriz columna para conv eficiente
    B, C, pH, pW = x_pad.shape
    cols = np.zeros((B, C, K, K, oH, oW), dtype=np.float32)
    for kh in range(K):
        for kw in range(K):
            cols[:, :, kh, kw, :, :] = x_pad[:, :, kh:kh+oH*S:S, kw:kw+oW*S:S]
    # → (B, C*K*K, oH*oW)
    return cols.reshape(B, C * K * K, oH * oW)


class Conv2d:
    def __init__(self, in_ch, out_ch, kernel=3, stride=1, padding=1, bias=True):
        self.in_ch, self.out_ch = in_ch, out_ch
        self.K, self.S, self.P = kernel, stride, padding
        self.use_bias = bias

        fan_in = in_ch * kernel * kernel
        self.W = kaiming_init(fan_in, (out_ch, in_ch, kernel, kernel))
        self.b = np.zeros(out_ch, dtype=np.float32) if bias else None
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b) if bias else None
        self._cache = None

    def forward(self, x):
        # x: (B, C, H, W)
        B, C, H, W = x.shape
        K, S, P = self.K, self.S, self.P
        oH = (H + 2*P - K) // S + 1
        oW = (W + 2*P - K) // S + 1

        xp = np.pad(x, ((0,0),(0,0),(P,P),(P,P)), mode='constant')
        col = _im2col(xp, K, S, oH, oW)          # (B, C*K*K, oH*oW)
        W_r = self.W.reshape(self.out_ch, -1)      # (out_ch, C*K*K)

        # (B, out_ch, oH*oW)
        out = np.tensordot(W_r, col, axes=([1],[1])).transpose(1,0,2)
        out = out.reshape(B, self.out_ch, oH, oW)

        if self.use_bias:
            out += self.b[None, :, None, None]

        self._cache = (x, xp, col)
        return out

    def backward(self, d_out):
        x, xp, col = self._cache
        B, C, H, W = x.shape
        K, S, P = self.K, self.S, self.P
        _, oC, oH, oW = d_out.shape

        # d_out: (B, out_ch, oH, oW) → (out_ch, B*oH*oW)
        d_out_r = d_out.transpose(1, 0, 2, 3).reshape(oC, -1)
        col_r   = col.transpose(1, 0, 2).reshape(C*K*K, -1)  #(C*K*K, B*oH*oW)

        self.dW += (d_out_r @ col_r.T / B).reshape(self.W.shape)
        if self.use_bias:
            self.db += d_out.sum(axis=(0,2,3)) / B

        W_r  = self.W.reshape(oC, -1) #(out_ch, C*K*K)
        dcol = W_r.T @ d_out_r #(C*K*K, B*oH*oW)
        dcol = dcol.reshape(C, K, K, B, oH, oW).transpose(3, 0, 1, 2, 4, 5)

        # col2im
        dxp = np.zeros_like(xp)
        for kh in range(K):
            for kw in range(K):
                dxp[:, :, kh:kh+oH*S:S, kw:kw+oW*S:S] += dcol[:, :, kh, kw, :, :]

        dx = dxp[:, :, P:P+H, P:P+W]
        return dx

    def parameters(self):
        params = [(self.W, self.dW)]
        if self.use_bias:
            params.append((self.b, self.db))
        return params

    def zero_grad(self):
        self.dW[:] = 0
        if self.use_bias:
            self.db[:] = 0


# BatchNorm2d para vectorizado por canal
class BatchNorm2d:
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        self.C, self.eps = num_features, eps
        self.momentum    = momentum
        self.gamma = np.ones(num_features,  dtype=np.float32)
        self.beta  = np.zeros(num_features, dtype=np.float32)
        self.dgamma = np.zeros_like(self.gamma)
        self.dbeta  = np.zeros_like(self.beta)
        self.running_mean = np.zeros(num_features, dtype=np.float32)
        self.running_var  = np.ones(num_features,  dtype=np.float32)
        self.training = True
        self._cache   = None

    def forward(self, x):
        #x: (B, C, H, W)
        B, C, H, W = x.shape
        if self.training:
            # Media y varianza por canal sobre (B, H, W)
            mean = x.mean(axis=(0,2,3))         
            var  = x.var(axis=(0,2,3))            
            self.running_mean = (1-self.momentum)*self.running_mean + self.momentum*mean
            self.running_var  = (1-self.momentum)*self.running_var  + self.momentum*var
        else:
            mean, var = self.running_mean, self.running_var

        std_inv = 1.0 / np.sqrt(var + self.eps)          
        xhat    = (x - mean[None,:,None,None]) * std_inv[None,:,None,None]
        out     = self.gamma[None,:,None,None]*xhat + self.beta[None,:,None,None]

        if self.training:
            self._cache = (x, xhat, mean, var, std_inv)
        return out

    def backward(self, d_out):
        x, xhat, mean, var, std_inv = self._cache
        B, C, H, W = x.shape
        N = B * H * W

        self.dgamma += (d_out * xhat).sum(axis=(0,2,3))
        self.dbeta  += d_out.sum(axis=(0,2,3))

        g = self.gamma[None,:,None,None]
        dxhat = d_out * g
        dvar  = (dxhat * (x - mean[None,:,None,None]) * (-0.5) *
                 (var + self.eps)[None,:,None,None]**(-1.5)).sum(axis=(0,2,3))
        dmean = ((-std_inv) * dxhat.sum(axis=(0,2,3)) +
                 dvar * (-2.0/N) * (x - mean[None,:,None,None]).sum(axis=(0,2,3)))

        dx = (dxhat * std_inv[None,:,None,None] +
              dvar[None,:,None,None] * 2*(x-mean[None,:,None,None])/N +
              dmean[None,:,None,None]/N)
        return dx

    def parameters(self):
        return [(self.gamma, self.dgamma), (self.beta, self.dbeta)]

    def zero_grad(self):
        self.dgamma[:] = 0
        self.dbeta[:] = 0



# Capas simples vectorizadas
class ReLU:
    def __init__(self):
        self._cache = None

    def forward(self, x):
        self._cache = x
        return np.maximum(0.0, x)

    def backward(self, d_out):
        return d_out * (self._cache > 0)

    def parameters(self):
        return []

    def zero_grad(self):
        pass


class MaxPool2d:
    def __init__(self, kernel=2, stride=2):
        self.K, self.S = kernel, stride
        self._cache = None

    def forward(self, x):
        B, C, H, W = x.shape
        K, S = self.K, self.S
        oH = (H - K) // S + 1
        oW = (W - K) // S + 1

        #Reshape a ventanas
        xr  = x[:, :, :oH*S, :oW*S]
        # (B, C, oH, S, oW, S)
        win = xr.reshape(B, C, oH, S, oW, S).transpose(0,1,2,4,3,5)
        win = win.reshape(B, C, oH, oW, K*K)# K=S aquí

        idx = win.argmax(axis=-1) #(B, C, oH, oW)
        out = win.max(axis=-1)
        self._cache = (x.shape, win, idx, K, S)
        return out

    def backward(self, d_out):
        x_shape, win, idx, K, S = self._cache
        B, C, H, W = x_shape
        oH, oW = d_out.shape[2], d_out.shape[3]

        dmask = np.zeros_like(win) # (B, C, oH, oW, K*K)
        np.put_along_axis(dmask, idx[..., None], d_out[..., None], axis=-1)

        dmask = dmask.reshape(B, C, oH, oW, K, K).transpose(0,1,2,4,3,5)
        dmask = dmask.reshape(B, C, oH*S, oW*S)

        dx = np.zeros(x_shape, dtype=np.float32)
        dx[:, :, :oH*S, :oW*S] = dmask
        return dx

    def parameters(self):
        return []

    def zero_grad(self):
        pass


class Upsample:
    def __init__(self, scale=2):
        self.scale = scale
        self._cache = None

    def forward(self, x):
        B, C, H, W = x.shape
        S = self.scale
        # Bilinear con NumPy
        oH, oW = H*S, W*S
        iy = (np.arange(oH, dtype=np.float32) + 0.5) / S - 0.5
        ix = (np.arange(oW, dtype=np.float32) + 0.5) / S - 0.5
        y0 = np.clip(np.floor(iy).astype(int), 0, H-1)
        x0 = np.clip(np.floor(ix).astype(int), 0, W-1)
        y1 = np.minimum(y0+1, H-1)
        x1 = np.minimum(x0+1, W-1)
        dy = (iy - y0)[:, None]   # (oH, 1)
        dx = (ix - x0)[None, :]   # (1, oW)

        out = (x[:,:,y0,:][:,:,:,x0]*(1-dy)*(1-dx) +
               x[:,:,y0,:][:,:,:,x1]*(1-dy)*dx      +
               x[:,:,y1,:][:,:,:,x0]*dy*(1-dx)       +
               x[:,:,y1,:][:,:,:,x1]*dy*dx)
        self._cache = (x.shape, y0, x0, y1, x1, dy, dx)
        return out

    def backward(self, d_out):
        x_shape, y0, x0, y1, x1, dy, dx = self._cache
        B, C, H, W = x_shape
        grad = np.zeros(x_shape, dtype=np.float32)
        np.add.at(grad, (slice(None), slice(None), y0[:,None], x0[None,:]), d_out*(1-dy)*(1-dx))
        np.add.at(grad, (slice(None), slice(None), y0[:,None], x1[None,:]), d_out*(1-dy)*dx)
        np.add.at(grad, (slice(None), slice(None), y1[:,None], x0[None,:]), d_out*dy*(1-dx))
        np.add.at(grad, (slice(None), slice(None), y1[:,None], x1[None,:]), d_out*dy*dx)
        return grad

    def parameters(self):
        return []

    def zero_grad(self):
        pass


class Concat:
    def __init__(self):
        self._cache = None

    def forward(self, a, b):
        self._cache = (a.shape[1], b.shape[1])
        return np.concatenate([a, b], axis=1)

    def backward(self, d_out):
        Ca, Cb = self._cache
        return d_out[:, :Ca], d_out[:, Ca:]

    def parameters(self):
        return []

    def zero_grad(self):
        pass


# Bloque ConvBnReLU
class ConvBnReLU:
    def __init__(self, in_ch, out_ch, kernel=3, padding=1):
        self.conv = Conv2d(in_ch, out_ch, kernel=kernel, padding=padding, bias=False)
        self.bn   = BatchNorm2d(out_ch)
        self.relu = ReLU()

    def forward(self, x):
        return self.relu.forward(self.bn.forward(self.conv.forward(x)))

    def backward(self, d_out):
        return self.conv.backward(self.bn.backward(self.relu.backward(d_out)))

    def parameters(self):
        return self.conv.parameters() + self.bn.parameters()

    def zero_grad(self):
        self.conv.zero_grad()
        self.bn.zero_grad()


# Optimizador SGD con Momentum
class SGD:
    def __init__(self, param_groups, lr=0.01, momentum=0.9, weight_decay=1e-4):
        #param_groups: lista de data_array & grad_array
        self.groups       = param_groups
        self.lr           = lr
        self.momentum     = momentum
        self.weight_decay = weight_decay
        self.velocity     = [np.zeros_like(p) for p, _ in param_groups]

    def step(self):
        for i, (p, g) in enumerate(self.groups):
            grad = g + self.weight_decay * p
            self.velocity[i] = self.momentum * self.velocity[i] - self.lr * grad
            p += self.velocity[i]

    def zero_grad(self):
        for _, g in self.groups:
            g[:] = 0.0



# SegNet-Lite para la red neuronal
class SegNetLite:
    #CNN Encoder-Decoder para segmentación semántica (NumPy).
    #Encoder: 3 etapas conv+pool  
    #Decoder: 3 etapas upsample+conv+skip
    
    def __init__(self, in_ch=3, num_classes=21):
        self.num_classes = num_classes

        self.enc1a = ConvBnReLU(in_ch, 16)
        self.enc1b = ConvBnReLU(16, 16)
        self.pool1 = MaxPool2d(2, 2)

        self.enc2a = ConvBnReLU(16, 32)
        self.enc2b = ConvBnReLU(32, 32)
        self.pool2 = MaxPool2d(2, 2)

        self.enc3a = ConvBnReLU(32, 64)
        self.enc3b = ConvBnReLU(64, 64)
        self.pool3 = MaxPool2d(2, 2)

        self.bot1 = ConvBnReLU(64, 128)
        self.bot2 = ConvBnReLU(128, 64)

        self.up3  = Upsample(2); self.cat3  = Concat()
        self.dec3a = ConvBnReLU(128, 64); self.dec3b = ConvBnReLU(64, 32)

        self.up2  = Upsample(2); self.cat2  = Concat()
        self.dec2a = ConvBnReLU(64, 32);  self.dec2b = ConvBnReLU(32, 16)

        self.up1  = Upsample(2); self.cat1  = Concat()
        self.dec1a = ConvBnReLU(32, 16);  self.dec1b = ConvBnReLU(16, 16)

        self.head = Conv2d(16, num_classes, kernel=1, padding=0)

        self._im = {}   # intermediates

    def forward(self, x):
        e1 = self.enc1b.forward(self.enc1a.forward(x))
        p1 = self.pool1.forward(e1)

        e2 = self.enc2b.forward(self.enc2a.forward(p1))
        p2 = self.pool2.forward(e2)

        e3 = self.enc3b.forward(self.enc3a.forward(p2))
        p3 = self.pool3.forward(e3)

        bt = self.bot2.forward(self.bot1.forward(p3))

        u3 = self.up3.forward(bt)
        c3 = self.cat3.forward(u3, e3)
        d3 = self.dec3b.forward(self.dec3a.forward(c3))

        u2 = self.up2.forward(d3)
        c2 = self.cat2.forward(u2, e2)
        d2 = self.dec2b.forward(self.dec2a.forward(c2))

        u1 = self.up1.forward(d2)
        c1 = self.cat1.forward(u1, e1)
        d1 = self.dec1b.forward(self.dec1a.forward(c1))

        logits = self.head.forward(d1)
        self._im = dict(e1=e1, e2=e2, e3=e3)
        return logits

    def backward(self, d_logits):
        #Backward parcial: cabeza + último bloque decoder
        dh = self.head.backward(d_logits)
        dh = self.dec1b.backward(dh)
        dh = self.dec1a.backward(dh)
        # (Backward completo omitido igual que en V1 — sólo se actualiza la cabeza+dec1)

    def predict(self, x):
        logits = self.forward(x)                          # (B, C, H, W)
        return np.argmax(logits, axis=1).astype(np.float32)  # (B, H, W)

    def parameters(self):
        #Devuelve lista de (data, grad) para el optimizador
        params = []
        layers = [self.enc1a, self.enc1b, self.enc2a, self.enc2b,
                  self.enc3a, self.enc3b, self.bot1, self.bot2,
                  self.dec3a, self.dec3b, self.dec2a, self.dec2b,
                  self.dec1a, self.dec1b, self.head]
        for l in layers:
            params.extend(l.parameters())
        return params

    def zero_grad(self):
        for _, g in self.parameters():
            g[:] = 0.0

    def save(self, path):
        data = {'__num_classes__': self.num_classes}
        for i, (p, _) in enumerate(self.parameters()):
            data[str(i)] = {'shape': list(p.shape), 'data': p.flatten().tolist()}
        with open(path, 'w') as f:
            json.dump(data, f)
        print(f"[Modelo] Guardado en {path} ({self.num_classes} clases)")

    def load(self, path):
        with open(path) as f:
            data = json.load(f)
        for i, (p, _) in enumerate(self.parameters()):
            entry = data.get(str(i))
            if entry:
                saved_shape = tuple(entry['shape'])
                if saved_shape != p.shape:
                    print(f"  aviso param {i}: forma guardada {saved_shape} != modelo {p.shape} — omitido")
                    continue
                p[:] = np.array(entry['data'], dtype=np.float32).reshape(p.shape)
        print(f"Modelo cargado desde {path}")



#Pérdida cross-entropy vectorizada
def segmentation_loss(logits, targets, ignore_index=255):
    #logits:  ndarray (B, C, H, W)
    #targets: ndarray (B, H, W) o (H, W)  con class_id enteros
    B, C, H, W = logits.shape

    # Asegurar (B, H, W)
    tgt = np.array(targets, dtype=np.int32)
    if tgt.ndim == 2:
        tgt = tgt[None]

    # Softmax estable a lo largo del eje de clases
    m      = logits.max(axis=1, keepdims=True)
    e      = np.exp(logits - m)
    probs  = e / e.sum(axis=1, keepdims=True)# (B,C,H,W)

    targets_int = tgt # (B,H,W)
    mask        = targets_int != ignore_index # (B,H,W) bool

    # Log-prob de la clase correcta
    b_idx = np.arange(B)[:, None, None]
    h_idx = np.arange(H)[None, :, None]
    w_idx = np.arange(W)[None, None, :]
    tc    = np.clip(targets_int, 0, C-1)  # evita OOB en pixels ignorados
    log_p = np.log(np.maximum(probs[b_idx, tc, h_idx, w_idx], 1e-9))

    count     = mask.sum()
    loss_mean = -(log_p * mask).sum() / max(count, 1)

    # Gradiente: softmax  one_hot  (solo en pixels válidos)
    d_logits = probs.copy()
    one_hot  = np.zeros_like(probs)
    one_hot[b_idx, tc, h_idx, w_idx] = 1.0
    d_logits -= one_hot
    d_logits *= mask[:, None, :, :]
    d_logits /= max(count, 1)

    return float(loss_mean), d_logits


# Lectura de imágenes 

def read_ppm(path):
    with open(path, 'rb') as f:
        magic = f.readline().decode().strip()
        assert magic == 'P6'
        while True:
            line = f.readline().decode().strip()
            if not line.startswith('#'): break
        w, h = map(int, line.split())
        maxval = int(f.readline().decode().strip())
        raw = f.read(w * h * 3)
    arr = np.frombuffer(raw, dtype=np.uint8).astype(np.float32)
    arr = (arr * 255 // maxval).astype(np.uint8)
    return w, h, arr.tolist()

def read_pgm(path):
    with open(path, 'rb') as f:
        magic = f.readline().decode().strip()
        assert magic == 'P5'
        while True:
            line = f.readline().decode().strip()
            if not line.startswith('#'): break
        w, h = map(int, line.split())
        _ = int(f.readline().decode().strip())
        raw = f.read(w * h)
    return w, h, list(raw)

def read_bmp(path):
    with open(path, 'rb') as f:
        data = f.read()
    assert data[:2] == b'BM'
    offset  = struct.unpack_from('<I', data, 10)[0]
    width   = struct.unpack_from('<i', data, 18)[0]
    height  = struct.unpack_from('<i', data, 22)[0]
    bpp     = struct.unpack_from('<H', data, 28)[0]
    assert bpp == 24
    flip    = height > 0
    height  = abs(height)
    row_size = (width * 3 + 3) & ~3
    pixels  = []
    for row in range(height):
        src_row = (height - 1 - row) if flip else row
        base    = offset + src_row * row_size
        for col in range(width):
            b_, g_, r_ = data[base+col*3], data[base+col*3+1], data[base+col*3+2]
            pixels.extend([r_, g_, b_])
    return width, height, pixels

def write_ppm(path, width, height, pixels):
    with open(path, 'wb') as f:
        f.write(f"P6\n{width} {height}\n255\n".encode())
        arr = np.array(pixels, dtype=np.uint8)
        arr = np.clip(arr, 0, 255)
        f.write(arr.tobytes())

def _read_png_minimal(path):
    import zlib
    with open(path, 'rb') as f:
        data = f.read()
    assert data[:8] == b'\x89PNG\r\n\x1a\n'
    pos = 8; width = height = bit_depth = color_type = 0; idat = []
    while pos < len(data):
        length     = struct.unpack_from('>I', data, pos)[0]
        chunk_type = data[pos+4:pos+8].decode('ascii', errors='replace')
        chunk_data = data[pos+8:pos+8+length]
        if chunk_type == 'IHDR':
            width = struct.unpack_from('>I', chunk_data, 0)[0]
            height = struct.unpack_from('>I', chunk_data, 4)[0]
            bit_depth = chunk_data[8]; color_type = chunk_data[9]
        elif chunk_type == 'IDAT': idat.append(chunk_data)
        elif chunk_type == 'IEND': break
        pos += 12 + length
    assert bit_depth == 8
    raw = zlib.decompress(b''.join(idat))
    ch = {0:1, 2:3, 3:1, 4:2, 6:4}.get(color_type, 3)
    stride = 1 + width * ch

    def paeth(a, b, c):
        p = a+b-c; pa,pb,pc = abs(p-a),abs(p-b),abs(p-c)
        return a if pa<=pb and pa<=pc else (b if pb<=pc else c)

    pixels = []; prev = bytes(width * ch)
    for y in range(height):
        base = y * stride; filt = raw[base]
        row  = bytearray(raw[base+1:base+1+width*ch])
        if filt==1:
            for i in range(ch, len(row)): row[i] = (row[i]+row[i-ch])&0xFF
        elif filt==2:
            for i in range(len(row)): row[i] = (row[i]+prev[i])&0xFF
        elif filt==3:
            for i in range(len(row)):
                a = row[i-ch] if i>=ch else 0
                row[i] = (row[i]+(a+prev[i])//2)&0xFF
        elif filt==4:
            for i in range(len(row)):
                a = row[i-ch] if i>=ch else 0
                b_ = prev[i]; c_ = prev[i-ch] if i>=ch else 0
                row[i] = (row[i]+paeth(a,b_,c_))&0xFF
        prev = bytes(row)
        for p in range(width):
            b0 = p*ch
            if ch == 1:   v = row[b0]; pixels.extend([v,v,v])
            elif ch == 2: v = row[b0]; pixels.extend([v,v,v])
            elif ch == 3: pixels.extend([row[b0],row[b0+1],row[b0+2]])
            else:         pixels.extend([row[b0],row[b0+1],row[b0+2]])
    return width, height, pixels

def read_mask_png(path):
    import zlib
    with open(path, 'rb') as f:
        data = f.read()
    assert data[:8] == b'\x89PNG\r\n\x1a\n'
    pos = 8; width = height = color_type = 0; idat = []
    while pos < len(data):
        length = struct.unpack_from('>I', data, pos)[0]
        ctype  = data[pos+4:pos+8].decode('ascii', errors='replace')
        cdata  = data[pos+8:pos+8+length]
        if ctype == 'IHDR':
            width = struct.unpack_from('>I', cdata, 0)[0]
            height = struct.unpack_from('>I', cdata, 4)[0]
            color_type = cdata[9]
        elif ctype == 'IDAT': idat.append(cdata)
        elif ctype == 'IEND': break
        pos += 12 + length
    raw = zlib.decompress(b''.join(idat))
    ch = 1 if color_type in (0,3) else (3 if color_type==2 else 4)
    stride = 1 + width*ch

    def paeth(a,b,c):
        p=a+b-c; pa,pb,pc=abs(p-a),abs(p-b),abs(p-c)
        return a if pa<=pb and pa<=pc else (b if pb<=pc else c)

    mask = []; prev = bytes(width*ch)
    for y in range(height):
        base = y*stride; filt = raw[base]
        row  = bytearray(raw[base+1:base+1+width*ch])
        if filt==1:
            for i in range(ch,len(row)): row[i]=(row[i]+row[i-ch])&0xFF
        elif filt==2:
            for i in range(len(row)): row[i]=(row[i]+prev[i])&0xFF
        elif filt==3:
            for i in range(len(row)):
                a=row[i-ch] if i>=ch else 0
                row[i]=(row[i]+(a+prev[i])//2)&0xFF
        elif filt==4:
            for i in range(len(row)):
                a=row[i-ch] if i>=ch else 0
                b_=prev[i]; c_=prev[i-ch] if i>=ch else 0
                row[i]=(row[i]+paeth(a,b_,c_))&0xFF
        prev = bytes(row)
        for p in range(width): mask.append(row[p*ch])
    return width, height, mask

def load_image_auto(path):
    ext = path.lower().rsplit('.', 1)[-1]
    if ext == 'ppm':  return read_ppm(path)
    if ext == 'pgm':
        w, h, d = read_pgm(path)
        px = []
        for v in d: px.extend([v,v,v])
        return w, h, px
    if ext == 'bmp':  return read_bmp(path)
    if ext == 'png':  return _read_png_minimal(path)
    if ext in ('jpg','jpeg'):
        if not HAS_CV2:
            raise ImportError("cv2 no disponible para leer JPEG")
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None: raise ValueError(f"No se pudo leer: {path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, _ = img.shape
        return w, h, img.flatten().tolist()
    raise ValueError(f"Formato '{ext}' no soportado.")



# Pre / post procesado  vectorizados

_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def preprocess(width, height, pixels, target_w=64, target_h=64):
    #Redimensiona (bilineal) y normaliza → ndarray (1, 3, H, W)
    img = np.array(pixels, dtype=np.float32).reshape(height, width, 3) / 255.0

    # Coordenadas fuente (bilinear)
    fy = (np.arange(target_h, dtype=np.float32) + 0.5) * height / target_h - 0.5
    fx = (np.arange(target_w, dtype=np.float32) + 0.5) * width  / target_w  - 0.5
    y0 = np.clip(np.floor(fy).astype(int), 0, height-1)
    x0 = np.clip(np.floor(fx).astype(int), 0, width-1)
    y1 = np.minimum(y0+1, height-1)
    x1 = np.minimum(x0+1, width-1)
    dy = (fy - y0)[:, None]     # (H, 1)
    dx = (fx - x0)[None, :]     # (1, W)

    resized = (img[y0[:,None], x0[None,:], :] * (1-dy)*(1-dx)[...,None] +
               img[y0[:,None], x1[None,:], :] * (1-dy)*dx[...,None]      +
               img[y1[:,None], x0[None,:], :] * dy*(1-dx)[...,None]       +
               img[y1[:,None], x1[None,:], :] * dy*dx[...,None]) # (H, W, 3)

    normalized = (resized - _MEAN) / _STD  # (H, W, 3)
    return normalized.transpose(2, 0, 1)[None] # (1, 3, H, W)

def preprocess_mask(mask_flat, orig_w, orig_h, target_w, target_h):
    #Nearest-neighbor resize de máscara → ndarray (1, H, W)
    arr = np.array(mask_flat, dtype=np.float32).reshape(1, orig_h, orig_w)
    sy = (np.arange(target_h) * orig_h // target_h).astype(int)
    sx = (np.arange(target_w) * orig_w // target_w).astype(int)
    return arr[:, sy[:, None], sx[None, :]]   # (1, target_h, target_w)

def resize_pred(pred, orig_w, orig_h):
    #pred: ndarray (B,H,W), float (B,orig_h,orig_w)
    _, pH, pW = pred.shape
    sy = (np.arange(orig_h) * pH // orig_h).astype(int)
    sx = (np.arange(orig_w) * pW // orig_w).astype(int)
    return pred[:, sy[:, None], sx[None, :]]

def colorize_mask(pred, width, height):
    #pred: ndarray (B,H,W) int → lista RGB plana para PPM.
    palette = get_palette()
    pal_arr = np.array([[r,g,b] for r,g,b,_ in palette], dtype=np.uint8)
    cls_map = np.clip(pred[0].astype(np.int32), 0, len(palette)-1)
    rgb     = pal_arr[cls_map]   # (H, W, 3)
    return rgb.flatten().tolist()

def tensor_to_image(pred):
    #pred: ndarray (B,H,W) → imagen BGR para cv2
    palette = get_palette()
    pal_arr = np.array([[r,g,b] for r,g,b,_ in palette], dtype=np.uint8)
    cls_map = np.clip(pred[0].astype(np.int32), 0, len(palette)-1)
    rgb = pal_arr[cls_map]
    return rgb[:,:,::-1]   # RGB→BGR para OpenCV



# Paletas semánticas pura visualizacion
ADE20K_CLASSES = [
    (120,120,120,"wall"),(180,120,120,"building"),(6,230,230,"sky"),
    (80,50,50,"floor"),(4,200,3,"tree"),(120,120,80,"ceiling"),
    (140,140,140,"road"),(204,5,255,"bed"),(230,230,230,"windowpane"),
    (4,250,7,"grass"),(224,5,255,"cabinet"),(235,255,7,"sidewalk"),
    (150,5,61,"person"),(120,120,70,"earth"),(8,255,51,"door"),
    (255,6,82,"table"),(143,255,140,"mountain"),(204,255,4,"plant"),
    (255,51,7,"curtain"),(204,70,3,"chair"),(0,102,200,"car"),
    (61,230,250,"water"),(255,6,51,"painting"),(11,102,255,"sofa"),
    (255,7,71,"shelf"),(255,9,224,"house"),(9,7,230,"sea"),
    (220,220,220,"mirror"),(255,9,92,"rug"),(112,9,255,"field"),
    (8,255,214,"armchair"),(7,255,224,"seat"),(255,184,6,"fence"),
    (10,255,71,"desk"),(255,41,10,"rock"),(7,255,255,"wardrobe"),
    (224,255,8,"lamp"),(102,8,255,"bathtub"),(255,61,6,"railing"),
    (255,194,7,"cushion"),(255,122,8,"base"),(0,255,20,"box"),
    (255,8,41,"column"),(255,5,153,"signboard"),(6,51,255,"chest"),
    (235,12,255,"counter"),(160,150,20,"sand"),(0,163,255,"sink"),
    (140,140,140,"skyscraper"),(250,10,15,"fireplace"),(20,255,0,"refrigerator"),
    (31,255,0,"grandstand"),(255,31,0,"path"),(255,224,0,"stairs"),
    (0,35,255,"runway"),(255,179,1,"case"),(0,255,245,"pool table"),
    (255,8,172,"pillow"),(0,255,112,"screen door"),(0,255,0,"stairway"),
    (0,143,255,"river"),(46,246,0,"bridge"),(255,98,0,"bookcase"),
    (0,255,153,"blind"),(255,56,0,"coffee table"),(255,0,51,"toilet"),
    (11,200,200,"flower"),(255,14,186,"book"),(100,0,255,"hill"),
    (0,163,255,"bench"),(255,10,39,"countertop"),(0,255,61,"stove"),
    (0,204,255,"palm"),(41,255,204,"kitchen island"),(0,255,204,"computer"),
    (0,51,100,"swivel chair"),(235,12,255,"boat"),(255,163,0,"bar"),
    (0,255,20,"arcade machine"),(0,92,255,"hovel"),(255,0,0,"bus"),
    (200,100,0,"towel"),(255,163,196,"light"),(0,152,255,"truck"),
    (255,0,255,"tower"),(0,255,255,"chandelier"),(255,204,204,"awning"),
    (255,153,0,"streetlight"),(0,255,10,"booth"),(255,0,153,"television"),
    (0,255,51,"airplane"),(0,184,255,"dirt track"),(0,214,255,"apparel"),
    (255,0,112,"pole"),(0,112,255,"land"),(0,74,255,"bannister"),
    (0,255,194,"escalator"),(255,122,0,"ottoman"),(0,255,92,"bottle"),
    (255,0,20,"buffet"),(255,255,0,"poster"),(0,255,133,"stage"),
    (255,0,92,"van"),(255,0,71,"ship"),(255,0,204,"fountain"),
    (0,102,100,"conveyer belt"),(0,255,0,"canopy"),(255,204,0,"washer"),
    (0,204,204,"plaything"),(255,51,255,"swimming pool"),(0,0,204,"stool"),
    (0,255,255,"barrel"),(204,255,204,"basket"),(0,102,255,"waterfall"),
    (255,0,163,"tent"),(255,163,51,"bag"),(0,82,255,"minibike"),
    (0,255,122,"cradle"),(255,0,255,"oven"),(255,71,0,"ball"),
    (0,0,255,"food"),(255,0,143,"step"),(163,255,0,"tank"),
    (0,255,0,"trade name"),(255,0,0,"microwave"),(255,255,204,"pot"),
    (0,163,0,"animal"),(0,255,255,"bicycle"),(255,0,163,"lake"),
    (163,0,255,"dishwasher"),(0,255,204,"screen"),(255,0,71,"blanket"),
    (0,255,163,"sculpture"),(0,0,255,"hood"),(255,255,0,"sconce"),
    (0,204,0,"vase"),(0,0,255,"traffic light"),(0,163,255,"tray"),
    (255,102,0,"ashcan"),(0,255,255,"fan"),(163,255,0,"pier"),
    (0,255,0,"crt screen"),(255,51,0,"plate"),(255,0,204,"monitor"),
    (0,255,153,"bulletin board"),(0,204,0,"shower"),(255,255,0,"radiator"),
    (0,163,255,"glass"),(255,153,0,"clock"),(0,204,255,"flag"),
]

COCO_CLASSES = [
    (0,0,0,"unlabeled"),(128,64,255,"person"),(0,128,128,"bicycle"),
    (255,128,0,"car"),(0,0,192,"motorcycle"),(128,255,0,"airplane"),
    (0,64,128,"bus"),(128,128,0,"train"),(64,0,128,"truck"),
    (192,128,0,"boat"),(64,128,0,"traffic light"),(0,192,128,"fire hydrant"),
    (128,0,64,"stop sign"),(0,64,192,"parking meter"),(192,0,64,"bench"),
    (64,128,128,"bird"),(0,192,64,"cat"),(128,192,0,"dog"),
    (64,64,0,"horse"),(192,64,0,"sheep"),(0,128,192,"cow"),
    (128,0,192,"elephant"),(64,0,192,"bear"),(0,64,64,"zebra"),
    (192,192,128,"giraffe"),(64,192,0,"backpack"),(128,64,0,"umbrella"),
    (0,128,64,"handbag"),(192,64,128,"tie"),(64,192,128,"suitcase"),
    (128,192,64,"frisbee"),(0,0,64,"skis"),(192,0,128,"snowboard"),
    (64,64,192,"sports ball"),(128,128,192,"kite"),(0,192,192,"baseball bat"),
    (192,128,192,"baseball glove"),(64,128,64,"skateboard"),(0,64,0,"surfboard"),
    (192,192,0,"tennis racket"),(128,0,0,"bottle"),(64,0,64,"wine glass"),
    (0,128,0,"cup"),(192,128,64,"fork"),(64,192,192,"knife"),
    (128,64,192,"spoon"),(0,192,0,"bowl"),(192,0,192,"banana"),
    (64,64,128,"apple"),(128,192,192,"sandwich"),(0,0,128,"orange"),
    (192,64,64,"broccoli"),(64,128,192,"carrot"),(128,0,128,"hot dog"),
    (0,64,255,"pizza"),(192,128,128,"donut"),(64,192,64,"cake"),
    (128,64,128,"chair"),(0,128,255,"couch"),(192,0,0,"potted plant"),
    (64,0,0,"bed"),(128,192,128,"dining table"),(0,192,255,"toilet"),
    (192,64,192,"tv"),(64,128,0,"laptop"),(128,128,128,"mouse"),
    (0,64,192,"remote"),(192,192,64,"keyboard"),(64,0,255,"cell phone"),
    (128,0,255,"microwave"),(0,128,255,"oven"),(192,64,255,"toaster"),
    (64,192,255,"sink"),(128,192,255,"refrigerator"),(0,0,255,"book"),
    (192,0,255,"clock"),(64,64,255,"vase"),(128,64,255,"scissors"),
    (0,192,255,"teddy bear"),(192,192,255,"hair drier"),(64,128,255,"toothbrush"),
    (0,64,0,"banner"),(128,128,64,"blanket"),(64,64,64,"bridge"),
    (0,192,64,"cardboard"),(128,0,64,"counter"),(64,128,64,"curtain"),
    (0,0,192,"dirt"),(128,192,64,"door-stuff"),(64,64,192,"fence"),
    (0,128,192,"floor-marble"),(128,64,192,"floor-other"),(64,0,64,"floor-stone"),
    (0,192,192,"floor-tile"),(128,192,192,"floor-wood"),(64,128,192,"grass"),
    (0,64,128,"gravel"),(128,0,128,"ceiling-other"),(64,192,128,"ceiling-tile"),
    (0,128,128,"rug"),(128,128,128,"sand"),(64,64,128,"sea"),
    (0,0,64,"snow"),(128,192,128,"sky-other"),(64,128,128,"sky-clouds"),
    (0,64,192,"wall-brick"),(128,0,192,"wall-concrete"),(64,192,192,"wall-other"),
    (0,128,64,"wall-panel"),(128,64,64,"wall-stone"),(64,0,128,"wall-tile"),
    (0,192,128,"wall-wood"),(128,192,64,"water-other"),(64,128,64,"waterdrops"),
    (0,64,64,"window-blind"),(128,0,64,"window-other"),(64,192,64,"tree"),
    (0,0,128,"roof"),(128,192,128,"rock"),(64,64,64,"sky"),
    (0,128,192,"playingfield"),(128,64,192,"pavement"),(64,0,192,"road"),
    (0,192,64,"mountain"),(128,192,192,"building-other"),(64,128,192,"platform"),
    (0,64,192,"light"),(128,0,192,"plant-other"),(64,192,128,"flower"),
    (0,128,128,"fog"),(128,128,64,"leaves"),(64,64,128,"net"),
    (0,0,192,"plastic"),(128,192,64,"railroad"),(64,128,64,"river"),
    (0,64,128,"salad"),(128,0,128,"sand2"),(64,192,128,"sea2"),
    (0,128,64,"shelf"),(128,64,64,"snow2"),(64,0,128,"solid-other"),
    (0,192,128,"stairs"),(128,192,128,"stone"),(64,128,128,"straw"),
    (0,64,64,"structural-other"),(128,0,64,"table2"),(64,192,64,"tent"),
    (0,0,64,"textile-other"),(128,192,192,"towel"),(64,64,192,"vegetable"),
    (0,128,192,"wall-other2"),(128,64,192,"water-other2"),(64,0,192,"waterfall"),
    (0,192,192,"wood"),(128,192,64,"other"),
]

DATASET_MODE = "ade20k"

def get_palette():
    return ADE20K_CLASSES if DATASET_MODE == "ade20k" else COCO_CLASSES

def get_num_classes():
    return len(get_palette())



# Dataset / descarga 

ADE20K_URL  = "http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip"
ADE20K_ROOT = "datasets/ade20k"
COCO_IMG_URL   = "http://images.cocodataset.org/zips/train2017.zip"
COCO_STUFF_URL = "http://calvin.inf.ed.ac.uk/wp-content/uploads/data/cocostuffdataset/stuffthingmaps_trainval2017.zip"
COCO_ROOT = "datasets/coco"

def _download(url, dest, desc=""):
    os.makedirs(os.path.dirname(dest) or '.', exist_ok=True)
    if os.path.exists(dest):
        print(f"  [cache] {os.path.basename(dest)}"); return
    print(f"  Descargando {desc or os.path.basename(dest)}")
    try:
        req = urllib.request.Request(url, headers={'User-Agent':'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=300) as r, open(dest,'wb') as f:
            total = int(r.headers.get('Content-Length',0)); done = 0
            while True:
                chunk = r.read(65536)
                if not chunk: break
                f.write(chunk); done += len(chunk)
                if total:
                    p = 100*done//total; bar = '█'*(p//5)+'░'*(20-p//5)
                    print(f"\r  [{bar}] {p}%  {done//1048576}MB", end='', flush=True)
        print()
    except urllib.error.URLError as e:
        print(f"\n  [ERROR] {e}"); raise

def _extract_zip(zp, dest):
    with zipfile.ZipFile(zp,'r') as z: z.extractall(dest)

def _extract_tar(tp, dest):
    with tarfile.open(tp,'r:gz') as t: t.extractall(dest)

def setup_ade20k(root=ADE20K_ROOT):
    data_dir = os.path.join(root, "ADEChallengeData2016")
    # Si los datos ya están extraídos los usamos directo
    if not os.path.isdir(data_dir):
        zip_p = os.path.join(root, "ADEChallengeData2016.zip")
        if os.path.exists(zip_p):
            print(f"  Extrayendo {os.path.basename(zip_p)} ...")
            _extract_zip(zip_p, root)
        else:
            _download(ADE20K_URL, zip_p, "ADE20K")
            _extract_zip(zip_p, root)
    img_dir  = os.path.join(data_dir, "images", "training")
    mask_dir = os.path.join(data_dir, "annotations", "training")
    if not os.path.isdir(img_dir):
        raise RuntimeError(
            f"No se encontró: {img_dir}\n"
            f"Estructura esperada: datasets/ade20k/ADEChallengeData2016/images/training/")
    samples = []
    for fn in sorted(os.listdir(img_dir)):
        if not fn.lower().endswith('.jpg'): continue
        mp = os.path.join(mask_dir, fn.replace('.jpg', '.png'))
        if os.path.exists(mp):
            samples.append((os.path.join(img_dir, fn), mp))
    print(f"[ADE20K] {len(samples):,} pares imagen/máscara encontrados")
    return samples


def setup_coco(root=COCO_ROOT):
    img_dir = os.path.join(root, "train2017")
    # Buscar máscaras en posibles ubicaciones (el zip puede extraerse diferente)
    mask_candidates = [
        os.path.join(root, "stuffthingmaps", "train2017"),
        os.path.join(root, "stuffthingmaps"),
        os.path.join(root, "annotations", "train2017"),
    ]
    mask_dir = next((d for d in mask_candidates if os.path.isdir(d)), None)
    # Imágenes
    if not os.path.isdir(img_dir):
        zip_p = os.path.join(root, "train2017.zip")
        if os.path.exists(zip_p):
            print(f"  Extrayendo {os.path.basename(zip_p)} ...")
            _extract_zip(zip_p, root)
        else:
            _download(COCO_IMG_URL, zip_p, "COCO images")
            _extract_zip(zip_p, root)
    # Máscaras
    if mask_dir is None:
        zip_p = os.path.join(root, "stuffthingmaps.zip")
        if os.path.exists(zip_p):
            print(f"  Extrayendo {os.path.basename(zip_p)} ...")
            _extract_zip(zip_p, root)
        else:
            _download(COCO_STUFF_URL, zip_p, "COCO-Stuff masks")
            _extract_zip(zip_p, root)
        mask_dir = next((d for d in mask_candidates if os.path.isdir(d)), None)
    if mask_dir is None:
        raise RuntimeError(f"No se encontró carpeta de máscaras COCO en {root}")
    print(f"[COCO] Máscaras: {mask_dir}")
    samples = []
    for fn in sorted(os.listdir(mask_dir)):
        if not fn.lower().endswith('.png'): continue
        ip = os.path.join(img_dir, fn.replace('.png', '.jpg'))
        if os.path.exists(ip):
            samples.append((ip, os.path.join(mask_dir, fn)))
    print(f"[COCO-Stuff] {len(samples):,} pares imagen/máscara encontrados")
    return samples

def _load_ade20k_pair(img_p, mask_p, tw, th):
    try:
        w, h, px = load_image_auto(img_p)
        mw, mh, mf = read_mask_png(mask_p)
    except Exception:
        return None, None
    mc = [255 if v==0 else min(v-1, len(ADE20K_CLASSES)-1) for v in mf]
    return preprocess(w, h, px, tw, th), preprocess_mask(mc, mw, mh, tw, th)

def _load_coco_pair(img_p, mask_p, tw, th):
    try:
        w, h, px = load_image_auto(img_p)
        mw, mh, mf = read_mask_png(mask_p)
    except Exception:
        return None, None
    n = len(COCO_CLASSES)
    mc = [255 if v==255 else min(v, n-1) for v in mf]
    return preprocess(w, h, px, tw, th), preprocess_mask(mc, mw, mh, tw, th)

def build_dataset(mode="both", target_w=64, target_h=64, max_samples=None, shuffle_=True):
    raw = []
    if mode in ("ade20k","both"):
        try:
            for p in setup_ade20k(): raw.append(("ade20k",p))
        except Exception as e: print(f"[ADE20K] {e}")
    if mode in ("coco","both"):
        try:
            for p in setup_coco(): raw.append(("coco",p))
        except Exception as e: print(f"[COCO] {e}")
    if shuffle_: random.shuffle(raw)
    if max_samples: raw = raw[:max_samples]
    print(f"\nPreprocesando {len(raw)} muestras ({target_w}×{target_h})...")
    dataset = []
    for i,(ds,(ip,mp)) in enumerate(raw):
        if i%100==0: print(f"  {i}/{len(raw)}...", end='\r', flush=True)
        if ds=="ade20k": inp,tgt = _load_ade20k_pair(ip,mp,target_w,target_h)
        else:            inp,tgt = _load_coco_pair(ip,mp,target_w,target_h)
        if inp is not None: dataset.append((inp,tgt,ds))
    print(f"\n  Muestras válidas: {len(dataset)}"); return dataset



# Entrenamiento
def _train_step(model, optimizer, inp, target):
    model.zero_grad()
    logits         = model.forward(inp)
    loss, d_logits = segmentation_loss(logits, target[0])   # target shape (1,H,W)
    model.backward(d_logits)
    optimizer.step()
    return loss

def train(mode="both", epochs=20, lr=0.005, target_w=64, target_h=64,
          max_samples=2000, checkpoint="pesos_segnet.json"):
    global DATASET_MODE
    print("═"*65)
    print("  SegNet-NumPy — Entrenamiento ADE20K + COCO")
    print("═"*65)
    dataset = build_dataset(mode=mode, target_w=target_w, target_h=target_h,
                            max_samples=max_samples)
    if not dataset:
        print("[ERROR] Sin muestras."); return None

    if mode == "ade20k":
        DATASET_MODE = "ade20k"; num_classes = len(ADE20K_CLASSES)
    elif mode == "coco":
        DATASET_MODE = "coco";   num_classes = len(COCO_CLASSES)
    else:
        DATASET_MODE = "ade20k"; num_classes = max(len(ADE20K_CLASSES), len(COCO_CLASSES))

    model     = SegNetLite(in_ch=3, num_classes=num_classes)
    total_p   = sum(p.size for p, _ in model.parameters())
    print(f"Parámetros: {total_p:,}")
    optimizer = SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    if os.path.exists(checkpoint):
        model.load(checkpoint)

    history = []; best_loss = math.inf
    for epoch in range(1, epochs+1):
        t0 = time.time(); total_loss = 0.0; n = 0
        random.shuffle(dataset)
        for i,(inp,tgt,ds) in enumerate(dataset):
            loss = _train_step(model, optimizer, inp, tgt)
            total_loss += loss; n += 1
            if (i+1)%50==0:
                print(f"  Época {epoch}/{epochs} | {i+1}/{len(dataset)} "
                      f"| loss={total_loss/n:.4f} | {time.time()-t0:.0f}s")
        avg = total_loss / max(n,1); elapsed = time.time()-t0
        print(f"\n[Época {epoch}/{epochs}] loss={avg:.4f} | {elapsed:.1f}s")
        history.append({'epoch':epoch,'loss':avg,'tiempo_s':elapsed})
        if epoch % 5 == 0:
            optimizer.lr *= 0.8
            print(f"  LR → {optimizer.lr:.5f}")
        if avg < best_loss:
            best_loss = avg; model.save(checkpoint)
            print(f"  Mejor guardado (loss={best_loss:.4f})")
    hist_path = checkpoint.replace('.json','_history.json')
    with open(hist_path,'w') as f: json.dump(history, f, indent=2)
    print(f"\nFinalizado. Mejor loss: {best_loss:.4f}")
    return model



# Inferencia sobre archivo
def segment_file(input_path, checkpoint="pesos_segnet.json",
                 output_path=None, target_w=64, target_h=64, mode="ade20k"):
    global DATASET_MODE
    DATASET_MODE = mode
    num_classes  = _load_num_classes(checkpoint)
    palette      = get_palette()
    if output_path is None:
        output_path = input_path.rsplit('.',1)[0] + '_seg.ppm'

    model = SegNetLite(in_ch=3, num_classes=num_classes)
    if os.path.exists(checkpoint):
        model.load(checkpoint)
        for attr in ['enc1a','enc1b','enc2a','enc2b','enc3a','enc3b',
                     'bot1','bot2','dec3a','dec3b','dec2a','dec2b','dec1a','dec1b']:
            getattr(model, attr).bn.training = False
    else:
        print(f"'{checkpoint}' no existe — pesos aleatorios.")

    print(f"Cargando {input_path}...")
    w, h, pixels = load_image_auto(input_path)
    inp  = preprocess(w, h, pixels, target_w, target_h)
    pred = model.predict(inp)          # (1, tH, tW)
    pred = resize_pred(pred, w, h)     # (1, h, w)
    write_ppm(output_path, w, h, colorize_mask(pred, w, h))
    print(f"Segmentación guardada: {output_path}")

    cnt = {}
    for v in pred[0].flatten().astype(int):
        cnt[v] = cnt.get(v, 0) + 1
    print(f"\nTop-10 clases detectadas:")
    for cls, n in sorted(cnt.items(), key=lambda x:-x[1])[:10]:
        if cls >= len(palette): continue
        name = palette[cls][3]; pct = 100*n/(w*h)
        print(f"  {name:22s} {'█'*int(pct/3):30s} {pct:5.1f}%")



# Cámara en tiempo real
def _load_num_classes(model_path):
    #Lee num_classes del checkpoint si existe, si no usa el default
    if model_path and os.path.exists(model_path):
        try:
            with open(model_path) as f:
                data = json.load(f)
            if '__num_classes__' in data:
                return int(data['__num_classes__'])
            # Inferirlo desde la forma del último parámetro (cabeza conv 1x1)
            last = data.get(str(max(int(k) for k in data if k.isdigit())))
            if last:
                return last['shape'][0]
        except Exception:
            pass
    return get_num_classes()

def realtime_camera(model_path=None):
    if not HAS_CV2:
        print("cv2 no disponible — instala opencv-python"); return
    num_classes = _load_num_classes(model_path)
    print(f"[Cámara] Cargando modelo con {num_classes} clases...")
    model = SegNetLite(in_ch=3, num_classes=num_classes)
    if model_path and os.path.exists(model_path):
        model.load(model_path)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("No se pudo abrir la cámara"); return
    print("Presiona 'q' para salir")
    while True:
        ret, frame = cap.read()
        if not ret: break
        h, w, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pixels    = frame_rgb.flatten().tolist()
        inp       = preprocess(w, h, pixels, target_w=64, target_h=64)
        pred      = model.predict(inp)
        pred_res  = resize_pred(pred, w, h)
        seg_img   = tensor_to_image(pred_res)
        cv2.imshow("Original", frame)
        cv2.imshow("Segmentacion", seg_img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release(); cv2.destroyAllWindows()



# Demo sin datos
def demo_sin_datos():
    global DATASET_MODE
    DATASET_MODE = "ade20k"
    W, H = 32, 32
    np.random.seed(0); random.seed(0)
    print("═"*65)
    print(" ADE20K/COCO simulado — demo sin descarga")
    print(f"  Python {sys.version.split()[0]} | NumPy {np.__version__}")
    print("═"*65)

    def synth_scene(seed):
        np.random.seed(seed)
        px   = np.zeros((H, W, 3), dtype=np.int32)
        mask = np.zeros((H, W),    dtype=np.int32)
        noise = np.random.randint(-15, 16, (H, W, 3))
        yf = np.arange(H) / H   # (H,)

        rows_sky   = yf < 0.30
        rows_wall  = (yf >= 0.30) & (yf < 0.65)
        rows_floor = (yf >= 0.65) & (yf < 0.80)
        rows_grass = yf >= 0.80

        px[rows_sky]   = np.clip(np.array([80, 140, 200])  + noise[rows_sky],   0, 255)
        px[rows_wall]  = np.clip(np.array([160, 155, 150]) + noise[rows_wall],  0, 255)
        px[rows_floor] = np.clip(np.array([100, 80,  60])  + noise[rows_floor], 0, 255)
        px[rows_grass] = np.clip(np.array([50,  120, 50])  + noise[rows_grass], 0, 255)

        mask[rows_sky]   = 2
        mask[rows_wall]  = 0
        mask[rows_floor] = 3
        mask[rows_grass] = 4

        return px.astype(np.uint8).flatten().tolist(), mask.flatten().tolist()

    print("\nGenerando escenas sintéticas...")
    dataset = []
    num_cls = len(ADE20K_CLASSES)
    for i in range(8):
        px, mf = synth_scene(i)
        inp = preprocess(W, H, px, W, H)
        tgt = np.array(mf, dtype=np.float32).reshape(1, H, W)
        dataset.append((inp, tgt, "ade20k"))

    px0, _ = synth_scene(0)
    write_ppm("demo_ade20k_entrada.ppm", W, H, px0)
    print("  demo_ade20k_entrada.ppm guardada")

    model = SegNetLite(in_ch=3, num_classes=num_cls)
    total = sum(p.size for p, _ in model.parameters())
    print(f"\nModelo CNN: {num_cls} clases | Parámetros: {total:,}")

    t0 = time.time()
    out = model.forward(dataset[0][0])
    ms  = (time.time()-t0)*1000
    loss0, _ = segmentation_loss(out, dataset[0][1][0])
    print(f"\nForward: {ms:.0f}ms  |  Loss inicial: {loss0:.4f}")

    print("\nEntrenando 3 pasos...")
    opt = SGD(model.parameters(), lr=0.01, momentum=0.9)
    for step in range(3):
        inp, tgt, _ = random.choice(dataset)
        t0  = time.time()
        lss = _train_step(model, opt, inp, tgt)
        print(f"  Paso {step+1}: loss={lss:.4f}  ({(time.time()-t0)*1000:.0f}ms)")

    pred = model.predict(dataset[0][0])
    write_ppm("demo_ade20k_salida.ppm", W, H, colorize_mask(pred, W, H))
    print("  demo_ade20k_salida.ppm guardada")

    cnt = {}
    for v in pred.flatten().astype(int):
        cnt[v] = cnt.get(v, 0) + 1
    print("\nClases ADE20K detectadas:")
    for cls, n in sorted(cnt.items(), key=lambda x:-x[1]):
        if cls >= len(ADE20K_CLASSES): continue
        name = ADE20K_CLASSES[cls][3]; pct = 100*n/(W*H)
        print(f"  {name:20s} {'█'*int(pct/4):20s} {pct:.1f}%")
    print("\n"+"═"*65)



# Main
if __name__ == "__main__":
    args = sys.argv[1:]

    def get_arg(flag, default=None):
        if flag in args:
            i = args.index(flag)
            if i + 1 < len(args): return args[i+1]
        return default

    if '--help' in args or '-h' in args:
        print("  --train      Entrenar y luego abrir cámara")
        print("  --camera     Solo cámara (requiere archivo entrenado)")
        print("  --predict    Segmentar imagen")
        print("  --demo       Demo sin datos")

    elif '--train' in args:
        checkpoint = get_arg('--checkpoint', 'pesos_segnet.json')
        train(mode=get_arg('--dataset','both'), epochs=int(get_arg('--epochs',20)),
              lr=float(get_arg('--lr',0.005)), target_w=int(get_arg('--res',64)),
              target_h=int(get_arg('--res',64)), max_samples=int(get_arg('--samples',2000)),
              checkpoint=checkpoint)
        realtime_camera(checkpoint)

    elif '--camera' in args:
        checkpoint = get_arg('--checkpoint', 'pesos_segnet.json')
        if not os.path.exists(checkpoint):
            print("No hay pesos. Ejecuta primero --train")
        else:
            realtime_camera(checkpoint)

    elif '--demo' in args:
        demo_sin_datos()

    elif '--predict' in args:
        img = get_arg('--input', '')
        if img and os.path.exists(img):
            segment_file(img, checkpoint=get_arg('--checkpoint','pesos_segnet.json'),
                         mode=get_arg('--dataset','ade20k'))
        else:
            print("Indica la imagen con --input ruta/imagen.jpg")

    else:
        print("NO ES POSIBLE EJECUTAR")
        print("Uso: python Version2_optimizada.py [--train|--camera|--predict|--demo]")
        print("  --train    Entrenar y abrir camara (--dataset --epochs --lr --res --samples --checkpoint)")
        print("  --camera   Solo camara (--checkpoint)")
        print("  --predict  Segmentar imagen (--input --checkpoint --dataset)")
        print("  --demo     Demo rapido sin internet")