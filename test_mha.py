import torch
from torch.nn import Linear, CrossEntropyLoss
from torch import nn
import torch.nn.functional as F
from transformers import BertPreTrainedModel, BertModel


def test_linear():
    linear1 = Linear(4, 2, bias=False)
    a = torch.randn(1, 4)
    b = linear1(a)
    print(b)


class ScaledDotProductAttention(nn.Module):
    """ Scaled Dot-Product Attention """

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1000)

        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)

        return output, attn


class MultiHeadAttention(nn.Module):
    """ Multi-Head Attention module """

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)

        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, q, k, v, mask=None):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual = q

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)  # For head axis broadcasting.

        q, attn = self.attention(q, k, v, mask=mask)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        q = self.dropout(self.fc(q))
        q += residual

        q = self.layer_norm(q)

        return q, attn


def seperate_seq(sequence_output, doc_len, ques_len, option_len):
    doc_seq_output = sequence_output.new(sequence_output.size()).zero_()
    doc_seq_mask = torch.zeros((sequence_output.size(0), sequence_output.size(1)))
    ques_seq_output = sequence_output.new(sequence_output.size()).zero_()
    ques_seq_mask = torch.zeros((sequence_output.size(0), sequence_output.size(1)))
    option_seq_output = sequence_output.new(sequence_output.size()).zero_()
    option_seq_mask = torch.zeros((sequence_output.size(0), sequence_output.size(1)))
    for i in range(doc_len.size(0)):
        doc_seq_output[i, :doc_len[i]] = sequence_output[i, 1:doc_len[i] + 1]
        doc_seq_mask[i, :doc_len[i]] = 1
        ques_seq_output[i, :ques_len[i]] = sequence_output[i, doc_len[i] + 1:doc_len[i] + ques_len[i] + 1]
        ques_seq_mask[i, :ques_len[i]] = 1
        option_seq_output[i, :option_len[i]] = sequence_output[i, doc_len[i] + ques_len[i] + 2:doc_len[i] + ques_len[i] + option_len[i] + 2]
        option_seq_mask[i, :option_len[i]] = 1
    return doc_seq_output, ques_seq_output, option_seq_output, doc_seq_mask, ques_seq_mask, option_seq_mask


def get_mask(hp_mask, hq_mask):
    a = hp_mask.unsqueeze(-1)
    b = hq_mask.unsqueeze(-2)
    mask_mat = torch.matmul(a, b)
    return mask_mat


class DUMA(BertPreTrainedModel):
    def __init__(self, config, n_layers=2, d_model=768, n_head=12, d_k=768, d_v=768, dropout=0.1):
        super(DUMA, self).__init__(config)
        self.num_choices = 15
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.layers = nn.ModuleList([nn.ModuleList([MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout),
                                                    MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)])
                                     for _ in range(n_layers)])
        self.fuse = nn.Linear(d_model * 2, 1)
        # self.classifier = nn.Linear(config.hidden_size, 1)
        self.classifier_all = nn.Linear(4, self.num_choices)
        self.init_weights()

    def forward(self, input_ids, token_type_ids, attention_mask, doc_len, ques_len, option_len, labels):
        flat_input_ids = input_ids.view(-1, input_ids.size(-1))  # [bs*4, sl]
        doc_len = doc_len.view(-1, doc_len.size(0) * doc_len.size(1)).squeeze()  # [1, bs*4]
        ques_len = ques_len.view(-1, ques_len.size(0) * ques_len.size(1)).squeeze()  # [1, bs*4]
        option_len = option_len.view(-1, option_len.size(0) * option_len.size(1)).squeeze()  # [1, bs*4]
        # print("input_ids:", input_ids.size())[1, 4, 512]
        flat_token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1))  # [bs*4, sl]
        flat_attention_mask = attention_mask.view(-1, attention_mask.size(-1))  # [bs*4, sl]
        sequence_output, pooled_output = self.bert(input_ids=flat_input_ids, token_type_ids=flat_token_type_ids, attention_mask=flat_attention_mask)[:2]

        doc_seq_output, ques_seq_output, option_seq_output, doc_seq_mask, ques_seq_mask, option_seq_mask = seperate_seq(sequence_output, doc_len, ques_len, option_len)
        hp, hq = ques_seq_output, option_seq_output
        pq_mask = get_mask(ques_seq_mask, option_seq_mask)
        qp_mask = get_mask(option_seq_mask, ques_seq_mask)
        for i, layer in enumerate(self.layers):
            hp, _ = layer[0](hp, hq, hq, mask=pq_mask)
            hq, _ = layer[1](hq, hp, hp, mask=qp_mask)

        hp, _ = hp.max(1)
        hq, _ = hq.max(1)
        cat_pq = torch.cat([hp, hq], dim=1)
        output_pool = self.dropout(cat_pq)
        match_logits = self.fuse(output_pool)
        match_reshaped_logits = match_logits.view(-1, 4)
        match_reshaped_logits = self.classifier_all(match_reshaped_logits)
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            match_loss = loss_fct(match_reshaped_logits, labels)
            return match_loss, match_reshaped_logits
        else:
            return match_reshaped_logits


def test_maxpool():
    # maxpool = nn.MaxPool1d(3, stride=3)
    a = torch.randn(1, 3, 4)
    b = torch.randn(1, 5, 4)
    # b = maxpool(a, )
    a, _ = a.max(dim=1)
    b, _ = b.max(dim=1)
    ab = torch.cat([a, b], dim=1)
    fuse = nn.Linear(4 * 2, 1)
    output = fuse(ab)

    print(output)


def test_mask():
    a = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 1]])
    b = torch.tensor([[1, 1, 0], [1, 0, 0]])
    a = a.unsqueeze(-1)
    b = b.unsqueeze(-2)
    print(a)
    print(b)
    mul = torch.matmul(a, b)
    print(mul.size())


def test_mha():
    torch.manual_seed(2)
    query = torch.randn(1, 3, 10)
    key = torch.randn(1, 3, 10)
    value = torch.randn(1, 3, 10)
    mha = MultiHeadAttention(2, 10, 11, 11)
    # mask = torch.tensor([[[[1,1, 1], [1,0,1],[0,0,1]],[[1,0, 0], [1,1,0],[0,0,1]]]], dtype=torch.long)
    mask = torch.tensor([[[1, 1, 1], [1, 0, 1], [0, 0, 1]]])
    b = mha(query, key, value, mask=mask)
    print(b[0])
    print(b[1])


def test_set_1():
    a = torch.randn(2, 3).zero_()
    a[1, :] = True
    print(a)


def test_duma():
    seq1 = [("中国是", "属于亚洲")] * 4
    seq2 = [("美国是一个", "发达国家")] * 4
    seqs = [seq1, seq2]
    from transformers import BertTokenizer, BertConfig
    tokenizer = BertTokenizer.from_pretrained("../dcmn/coin/c3pretrain/chinese-bert-wwm-ext/")
    input_ids_all = []
    token_type_ids_all = []
    attention_mask_all = []
    for seq_i in seqs:
        input_ids = []
        token_type_ids = []
        attention_mask = []
        for seq in seq_i:
            tokens = tokenizer.encode_plus(seq[0], seq[1], max_length=16, pad_to_max_length=True)
            input_ids.append(tokens['input_ids'])
            token_type_ids.append(tokens['token_type_ids'])
            attention_mask.append(tokens['attention_mask'])
        input_ids_all.append(input_ids)
        token_type_ids_all.append(token_type_ids)
        attention_mask_all.append(attention_mask)

    config = BertConfig.from_pretrained("../dcmn/coin/c3pretrain/chinese-bert-wwm-ext/")
    model = DUMA(config=config)
    input_ids_all = torch.tensor(input_ids_all, dtype=torch.long)
    token_type_ids_all = torch.tensor(token_type_ids_all, dtype=torch.long)
    attention_mask_all = torch.tensor(attention_mask_all, dtype=torch.long)
    doc_len = torch.tensor([[0, 0, 0, 0], [0, 0, 0, 0]], dtype=torch.long)
    ques_len = torch.tensor([[3, 3, 3, 3], [5, 5, 5, 5]], dtype=torch.long)
    option_len = torch.tensor([[4, 4, 4, 4], [4, 4, 4, 4]], dtype=torch.long)
    labels = torch.tensor([1, 2], dtype=torch.long)
    output = model(input_ids_all, token_type_ids_all, attention_mask_all, doc_len, ques_len, option_len, labels)
    print(output)

def test_cpu_gpu():
    a = torch.tensor([1,2,3])
    a = a.cuda()
    print(a.device)

if __name__ == '__main__':
    # test_linear()
    # test_maxpool()
    test_mha()
    # test_mask()
    # test_set_1()
    # test_duma()
    # print(test_cpu_gpu())