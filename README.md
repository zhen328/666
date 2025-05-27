使用 LSTM 的序列到序列学习（Seq2Seq） 1. 环境设置与库导入 python   运行      import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.legacy.data import Field, Dataset, BucketIterator
from torchtext.legacy.datasets import Multi30k
import random
import math
import numpy as np

# 设置随机种子
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

# 检查是否可用 GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")
      2. 数据预处理 2.1 加载数据集（以机器翻译为例） python   运行      # 定义字段（Field）：处理源语言（德语）和目标语言（英语）
SRC = Field(tokenize='spacy', 
           tokenizer_language='de_core_news_sm', 
           init_token='<sos>', 
           eos_token='<eos>', 
           lower=True)

TRG = Field(tokenize='spacy', 
           tokenizer_language='en_core_web_sm', 
           init_token='<sos>', 
           eos_token='<eos>', 
           lower=True)

# 加载 Multi30k 数据集（德语→英语）
train_data, valid_data, test_data = Multi30k.splits(exts=('.de', '.en'), 
                                                   fields=(SRC, TRG))
      2.2 构建词汇表 python   运行      # 构建词汇表（限制词汇量，添加未知词 <unk>）
SRC.build_vocab(train_data, min_freq=2)
TRG.build_vocab(train_data, min_freq=2)

print(f"源语言词汇表大小: {len(SRC.vocab)}")
print(f"目标语言词汇表大小: {len(TRG.vocab)}")
      2.3 创建数据迭代器 python   运行      # 创建数据加载器（BucketIterator 自动按长度排序，减少填充）
BATCH_SIZE = 128

train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
    (train_data, valid_data, test_data),
    batch_size=BATCH_SIZE,
    device=device)
      3. 定义 Seq2Seq 模型 3.1 编码器（Encoder） python   运行      class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        
        self.input_dim = input_dim
        self.emb_dim = emb_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        
        # 嵌入层
        self.embedding = nn.Embedding(input_dim, emb_dim)
        
        # LSTM 层
        self.lstm = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout)
        
        #  dropout 层
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, src):
        # src: [src_len, batch_size]
        
        # 嵌入层
        embedded = self.dropout(self.embedding(src))  # [src_len, batch_size, emb_dim]
        
        # LSTM 层
        outputs, (hidden, cell) = self.lstm(embedded)  # outputs: [src_len, batch_size, hid_dim]
                                                        # hidden/cell: [n_layers, batch_size, hid_dim]
        
        return hidden, cell
      3.2 解码器（Decoder，带注意力机制） python   运行      class Attention(nn.Module):
    def __init__(self, hid_dim):
        super().__init__()
        
        self.hid_dim = hid_dim
        
        # 注意力层：计算注意力权重
        self.attn = nn.Linear(2 * hid_dim, hid_dim)
        self.v = nn.Linear(hid_dim, 1, bias=False)
    
    def forward(self, hidden, encoder_outputs):
        # hidden: [n_layers, batch_size, hid_dim]（当前解码器隐藏状态）
        # encoder_outputs: [src_len, batch_size, hid_dim]（所有编码器输出）
        
        batch_size = hidden.shape[1]
        src_len = encoder_outputs.shape[0]
        
        # 重复 hidden 到 src_len 次，便于计算每个位置的注意力
        hidden = hidden[-1].unsqueeze(0).repeat(src_len, 1, 1)  # [src_len, batch_size, hid_dim]
        encoder_outputs = encoder_outputs.permute(1, 0, 2)      # [batch_size, src_len, hid_dim]
        
        # 拼接 hidden 和 encoder_outputs
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))  # [batch_size, src_len, hid_dim]
        
        # 计算注意力权重
        attention = self.v(energy).squeeze(2)  # [batch_size, src_len]
        return F.softmax(attention, dim=1)     # 归一化权重
      python   运行      class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout, attention):
        super().__init__()
        
        self.output_dim = output_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.attention = attention
        
        # 嵌入层
        self.embedding = nn.Embedding(output_dim, emb_dim)
        
        # LSTM 层
        self.lstm = nn.LSTM(emb_dim + hid_dim, hid_dim, n_layers, dropout=dropout)
        
        # 输出层
        self.out = nn.Linear(emb_dim + 2 * hid_dim, output_dim)
        
        #  dropout 层
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, input_token, hidden, cell, encoder_outputs):
        # input_token: [batch_size]（当前输入的目标词）
        # hidden/cell: [n_layers, batch_size, hid_dim]（解码器隐藏状态）
        # encoder_outputs: [src_len, batch_size, hid_dim]（所有编码器输出）
        
        input_token = input_token.unsqueeze(0)  # [1, batch_size]
        
        # 嵌入层
        embedded = self.dropout(self.embedding(input_token))  # [1, batch_size, emb_dim]
        
        # 计算注意力权重
        attn_weights = self.attention(hidden, encoder_outputs)  # [batch_size, src_len]
        attn_weights = attn_weights.unsqueeze(1)  # [batch_size, 1, src_len]
        
        # 加权求和得到上下文向量
        encoder_outputs = encoder_outputs.permute(1, 0, 2)  # [batch_size, src_len, hid_dim]
        context = torch.bmm(attn_weights, encoder_outputs)  # [batch_size, 1, hid_dim]
        context = context.permute(1, 0, 2)  # [1, batch_size, hid_dim]
        
        # 拼接嵌入向量和上下文向量
        lstm_input = torch.cat((embedded, context), dim=2)  # [1, batch_size, emb_dim+hid_dim]
        
        # 解码器 LSTM 层
        output, (hidden, cell) = self.lstm(lstm_input, (hidden, cell))  # output: [1, batch_size, hid_dim]
        
        # 拼接嵌入向量、上下文向量和解码器输出
        embedded = embedded.squeeze(0)  # [batch_size, emb_dim]
        output = output.squeeze(0)      # [batch_size, hid_dim]
        context = context.squeeze(0)    # [batch_size, hid_dim]
        
        # 输出层
        pred = self.out(torch.cat((embedded, output, context), dim=1))  # [batch_size, output_dim]
        
        return pred, hidden, cell
      3.3 序列到序列模型（Seq2Seq） python   运行      class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        
    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        # src: [src_len, batch_size]
        # trg: [trg_len, batch_size]（包含 <sos> 和 <eos>）
        # teacher_forcing_ratio: 教师强制比例（0~1）
        
        batch_size = src.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim
        
        # 初始化输出张量（存储每个时间步的预测结果）
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)
        
        # 编码器输出
        hidden, cell = self.encoder(src)
        
        # 解码器初始输入为 <sos> 标记
        input_token = trg[0, :]  # [batch_size]
        
        for t in range(1, trg_len):
            # 解码器前向传播
            output, hidden, cell = self.decoder(input_token, hidden, cell, encoder_outputs=hidden)
            
            # 存储预测结果
            outputs[t] = output
            
            # 教师强制：随机选择是否使用真实目标词作为下一个输入
            teacher_force = random.random() < teacher_forcing_ratio
            input_token = trg[t] if teacher_force else output.argmax(1)
        
        return outputs
      4. 模型训练与评估 4.1 超参数设置 python   运行      # 超参数
INPUT_DIM = len(SRC.vocab)
OUTPUT_DIM = len(TRG.vocab)
EMB_DIM = 256
HID_DIM = 512
N_LAYERS = 2
ENC_DROPOUT = 0.5
DEC_DROPOUT = 0.5

# 初始化注意力机制和模型
attention = Attention(HID_DIM)
encoder = Encoder(INPUT_DIM, EMB_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT).to(device)
decoder = Decoder(OUTPUT_DIM, EMB_DIM, HID_DIM, N_LAYERS, DEC_DROPOUT, attention).to(device)
model = Seq2Seq(encoder, decoder, device).to(device)
      4.2 损失函数与优化器 python   运行      # 损失函数（忽略 <pad> 标记）
PAD_IDX = TRG.vocab.stoi['<pad>']
criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX).to(device)

# 优化器
optimizer = optim.Adam(model.parameters())
      4.3 训练循环 python   运行      def train(model, iterator, optimizer, criterion, clip):
    model.train()
    total_loss = 0
    
    for i, batch in enumerate(iterator):
        src = batch.src.transpose(0, 1)  # [src_len, batch_size]
        trg = batch.trg.transpose(0, 1)  # [trg_len, batch_size]
        
        optimizer.zero_grad()
        outputs = model(src, trg)  # [trg_len, batch_size, output_dim]
        
        # 调整维度以计算损失
        outputs = outputs[1:].view(-1, OUTPUT_DIM)
        trg = trg[1:].view(-1)
        
        loss = criterion(outputs, trg)
        loss.backward()
        
        # 梯度裁剪（防止梯度爆炸）
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        
        optimizer.step()
        total_loss += loss.item()
    
    return total_loss / len(iterator)
      4.4 评估循环 python   运行      def evaluate(model, iterator, criterion):
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch in iterator:
            src = batch.src.transpose(0, 1)
            trg = batch.trg.transpose(0, 1)
            
            outputs = model(src, trg, teacher_forcing_ratio=0)  # 评估时不使用教师强制
            
            outputs = outputs[1:].view(-1, OUTPUT_DIM)
            trg = trg[1:].view(-1)
            
            loss = criterion(outputs, trg)
            total_loss += loss.item()
    
    return total_loss / len(iterator)
      4.5 训练过程 python   运行      # 训练参数
N_EPOCHS = 10
CLIP = 1

best_valid_loss = float('inf')

for epoch in range(N_EPOCHS):
    train_loss = train(model, train_iterator, optimizer, criterion, CLIP)
    valid_loss = evaluate(model, valid_iterator, criterion)
    
    print(f"Epoch: {epoch+1:02}")
    print(f"  训练损失: {train_loss:.3f}")
    print(f"  验证损失: {valid_loss:.3f}")
    
    # 保存最优模型
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'best_model.pth')
      5. 模型推理与测试 python   运行      def translate_sentence(sentence, src_field, trg_field, model, device, max_len=50):
    model.eval()
    
    # 预处理输入句子
    tokenized = [tok.text for tok in src_field.tokenize(sentence)]
    tokenized = [src_field.init_token] + tokenized + [src_field.eos_token]
    src_indices = [src_field.vocab.stoi[tok] for tok in tokenized]
    src_tensor = torch.LongTensor(src_indices).unsqueeze(1).to(device)
    
    # 编码器输出
    with torch.no_grad():
        hidden, cell = model.encoder(src_tensor)
    
    # 解码器初始输入为 <sos>
    trg_indices = [trg_field.vocab.stoi[trg_field.init_token]]
    for _ in range(max_len):
        trg_tensor = torch.LongTensor([trg_indices[-1]]).to(device)
        
        with torch.no_grad():
            output, hidden, cell = model.decoder(trg_tensor, hidden, cell, model.encoder(src_tensor))
        
        pred_token = output.argmax(1).item()
        trg_indices.append(pred_token)
        
        if pred_token == trg_field.vocab.stoi[trg_field.eos_token]:
            break
    
    # 转换为单词
    trg_tokens = [trg_field.vocab.itos[i] for i in trg_indices]
    return trg_tokens[1:-1]  # 移除 <sos> 和 <eos>
      测试示例 python   运行      # 示例输入（德语句子）
example_german = "ein mann mit einem rucksack steht an einem ufer."

# 翻译
translated = translate_sentence(
    example_german, SRC, TRG, model, device, max_len=50
)

print(f"德语原文: {example_german}")
print(f"英语翻译: {' '.join(translated)}")
      关键说明 1.  编码器 - 解码器架构： ◦ 编码器将输入序列编码为固定长度的隐藏状态。 ◦ 解码器利用编码器的隐藏状态逐词生成目标序列。   2.  注意力机制： ◦ 允许解码器在生成每个词时关注输入序列的不同部分，提升长序列翻译效果。 ◦ 通过 Attention 类计算注意力权重，结合上下文向量优化预测。   3.  教师强制（Teacher Forcing）： ◦ 训练时随机使用真实目标词作为输入，加速收敛并减少曝光偏差（Exposure Bias）。   4.  梯度裁剪（Gradient Clipping）： ◦ 防止反向传播时梯度爆炸，提高训练稳定性。    扩展建议 • 数据增强：使用回译（Back Translation）、同义词替换等方法扩充训练数据。 • 模型优化：尝试双向 LSTM、多层 LSTM 或更换为 Transformer 架构。 • 超参数调优：调整嵌入维度（EMB_DIM）、隐藏层维度（HID_DIM）、批次大小等。 • 可视化：使用 TensorBoard 监控训练过程，或可视化注意力权重分布。  运行此 Notebook 前需安装依赖： bash        pip install torch torchtext spacy
python -m spacy download de_core_news_sm en
