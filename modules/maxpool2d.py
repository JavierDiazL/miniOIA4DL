from modules.layer import Layer
#from cython_modules.maxpool2d import maxpool_forward_cython
import numpy as np

class MaxPool2D(Layer):
    def __init__(self, kernel_size, stride):
        self.kernel_size = kernel_size
        self.stride = stride

    # def forward(self, input, training=True):  # input: np.ndarray of shape [B, C, H, W]
    #     self.input = input
    #     B, C, H, W = input.shape
    #     KH, KW = self.kernel_size, self.kernel_size
    #     SH, SW = self.stride, self.stride

    #     out_h = (H - KH) // SH + 1
    #     out_w = (W - KW) // SW + 1

    #     self.max_indices = np.zeros((B, C, out_h, out_w, 2), dtype=int)
    #     output = np.zeros((B, C, out_h, out_w), dtype=input.dtype)

    #     for b in range(B):
    #         for c in range(C):
    #             for i in range(out_h):
    #                 for j in range(out_w):
    #                     h_start = i * SH
    #                     h_end = h_start + KH
    #                     w_start = j * SW
    #                     w_end = w_start + KW

    #                     window = input[b, c, h_start:h_end, w_start:w_end]
    #                     max_idx = np.unravel_index(np.argmax(window), window.shape)
    #                     max_val = window[max_idx]

    #                     output[b, c, i, j] = max_val
    #                     self.max_indices[b, c, i, j] = (h_start + max_idx[0], w_start + max_idx[1])

    #     return output

    def forward(self, input, training=True):
        self.input = input
        B, C, H, W = input.shape
        KH, KW = self.kernel_size, self.kernel_size
        SH, SW = self.stride, self.stride

        out_h = (H - KH) // SH + 1
        out_w = (W - KW) // SW + 1
        
        # --- INICIO BLOQUE GENERADO CON IA ---
        output = np.full((B, C, out_h, out_w), -np.inf, dtype=input.dtype)

        # En lugar de iterar i,j → iteramos sobre KH,KW (mucho menos iteraciones)
        # KH,KW suele ser 2x2 o 3x3 → solo 4 o 9 iteraciones en lugar de out_h*out_w
        for kh in range(KH):
            for kw in range(KW):
                # Selecciona de golpe todos los píxeles afectados por esta posición del kernel
                # para TODOS los (b, c, i, j) a la vez
                rows = slice(kh, kh + out_h * SH, SH)
                cols = slice(kw, kw + out_w * SW, SW)
                patch = input[:, :, rows, cols]   # shape: (B, C, out_h, out_w)

                # Actualiza el máximo elemento a elemento
                np.maximum(output, patch, out=output)

        return output
         # --- FIN BLOQUE GENERADO CON IA ---

    def backward(self, grad_output, learning_rate=None):
        B, C, H, W = self.input.shape
        grad_input = np.zeros_like(self.input, dtype=grad_output.dtype)
        out_h, out_w = grad_output.shape[2], grad_output.shape[3]

        for b in range(B):
            for c in range(C):
                for i in range(out_h):
                    for j in range(out_w):
                        r, s = self.max_indices[b, c, i, j]
                        grad_input[b, c, r, s] += grad_output[b, c, i, j]

        return grad_input