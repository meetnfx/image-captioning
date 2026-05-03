import matplotlib.pyplot as plt

epochs = range(1, 11)

# baseline
lstm_train = [3.21, 2.85, 2.77, 2.72, 2.69, 2.67, 2.66, 2.65, 2.64, 2.62]
lstm_val   = [2.68, 2.58, 2.54, 2.51, 2.50, 2.49, 2.48, 2.47, 2.47, 2.46]

#attent
attn_train = [2.98, 2.49, 2.38, 2.30, 2.25, 2.21, 2.18, 2.15, 2.13, 2.11]
attn_val   = [2.56, 2.44, 2.39, 2.36, 2.35, 2.34, 2.33, 2.33, 2.33, 2.33]

# transformer
trans_train = [3.22, 2.61, 2.46, 2.36, 2.29, 2.24, 2.19, 2.15, 2.12, 2.09]
trans_val   = [2.68, 2.49, 2.40, 2.34, 2.31, 2.28, 2.26, 2.25, 2.24, 2.23]



fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
c_lstm = '#7f8c8d'  
c_attn = '#3498db' 
c_trans = '#e67e22'  

ax1.plot(epochs, lstm_train, label='Baseline LSTM', color=c_lstm, marker='o', linewidth=2)
ax1.plot(epochs, attn_train, label='Spatial Attention', color=c_attn, marker='s', linewidth=2)
ax1.plot(epochs, trans_train, label='Transformer', color=c_trans, marker='^', linewidth=2)
ax1.set_title('Training Loss Comparison', fontsize=14, pad=10)
ax1.set_xlabel('Epochs', fontsize=12)
ax1.set_ylabel('Cross Entropy Loss', fontsize=12)
ax1.grid(True, linestyle='--', alpha=0.6)
ax1.legend(fontsize=11)
ax1.set_xticks(epochs)
ax2.plot(epochs, lstm_val, label='Baseline LSTM', color=c_lstm, marker='o', linewidth=2)
ax2.plot(epochs, attn_val, label='Spatial Attention', color=c_attn, marker='s', linewidth=2)
ax2.plot(epochs, trans_val, label='Transformer', color=c_trans, marker='^', linewidth=2)
ax2.set_title('Validation Loss Comparison', fontsize=14, pad=10)
ax2.set_xlabel('Epochs', fontsize=12)
ax2.grid(True, linestyle='--', alpha=0.6)
ax2.legend(fontsize=11)
ax2.set_xticks(epochs)

plt.tight_layout()
save_path = 'combined_loss_curves.png'
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"Success! Saved publication-ready graph to {save_path}")