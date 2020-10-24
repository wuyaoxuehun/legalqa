import torch
from torch.autograd import Variable
from torch.nn import Linear, CrossEntropyLoss
from torch import nn
import torch.nn.functional as F
from transformers import BertPreTrainedModel, BertModel


class ScaledDotProductAttention(nn.Module):
    """ Scaled Dot-Product Attention """

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -10000)

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
        # self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, q, k, v, mask=None):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        # residual = q

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
        # q += residual
        # q = self.layer_norm(q)
        return q, attn


def seperate_seq(sequence_output, doc_len, ques_len, option_len):
    doc_seq_output = sequence_output.new(sequence_output.size()).zero_()
    doc_seq_mask = torch.zeros((sequence_output.size(0), sequence_output.size(1)), device=sequence_output.device)
    ques_seq_output = sequence_output.new(sequence_output.size()).zero_()
    ques_seq_mask = torch.zeros((sequence_output.size(0), sequence_output.size(1)), device=sequence_output.device)
    option_seq_output = sequence_output.new(sequence_output.size()).zero_()
    option_seq_mask = torch.zeros((sequence_output.size(0), sequence_output.size(1)), device=sequence_output.device)
    for i in range(doc_len.size(0)):
        doc_seq_output[i, :doc_len[i]] = sequence_output[i, 1:doc_len[i] + 1]
        doc_seq_mask[i, :doc_len[i]] = 1
        ques_seq_output[i, :ques_len[i]] = sequence_output[i, doc_len[i] + 1:doc_len[i] + ques_len[i] + 1]
        ques_seq_mask[i, :ques_len[i]] = 1
        option_seq_output[i, :option_len[i]] = sequence_output[i, doc_len[i] + ques_len[i] + 2:doc_len[i] + ques_len[i] + option_len[i] + 2]
        option_seq_mask[i, :option_len[i]] = 1
    return doc_seq_output, ques_seq_output, option_seq_output, doc_seq_mask, ques_seq_mask, option_seq_mask


def masked_softmax(vector, seq_lens):
    mask = vector.new(vector.size()).zero_()
    for i in range(seq_lens.size(0)):
        mask[i, :, :seq_lens[i]] = 1
    mask = Variable(mask, requires_grad=False)
    # mask = None
    if mask is None:
        result = torch.nn.functional.softmax(vector, dim=-1)
    else:
        result = torch.nn.functional.softmax(vector * mask, dim=-1)
        result = result * mask
        result = result / (result.sum(dim=-1, keepdim=True) + 1e-13)
    return result


class FuseNet(nn.Module):
    def __init__(self, config):
        super(FuseNet, self).__init__()
        self.linear1 = nn.Linear(config.hidden_size, config.hidden_size)
        self.linear2 = nn.Linear(config.hidden_size, config.hidden_size)

    def forward(self, inputs):
        p, q = inputs
        lq = self.linear1(q)
        lp = self.linear2(p)
        mid = nn.Sigmoid()(lq + lp)
        output = p * mid + q * (1 - mid)
        return output


class SingleMatchNet(nn.Module):
    def __init__(self, config):
        super(SingleMatchNet, self).__init__()
        self.trans_linear = nn.Linear(config.hidden_size, config.hidden_size)
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)

    def forward(self, inputs):
        proj_p, proj_q, seq_len = inputs
        trans_q = self.trans_linear(proj_q)
        att_weights = proj_p.bmm(torch.transpose(trans_q, 1, 2))
        att_norm = masked_softmax(att_weights, seq_len)
        att_vec = att_norm.bmm(proj_q)
        output = nn.ReLU()(self.dense(att_vec))
        return output


def get_mask(hp_mask, hq_mask):
    a = hp_mask.unsqueeze(-1)
    b = hq_mask.unsqueeze(-2)
    mask_mat = torch.matmul(a, b)
    return mask_mat


class AOI(nn.Module):
    def __init__(self, config):
        super(AOI, self).__init__()
        self.trans_linear = nn.Linear(config.hidden_size, config.hidden_size)
        self.dense3 = nn.Linear(3 * config.hidden_size, config.hidden_size)
        self.fusechoice = FuseNet(config)
        self.choice_match = SingleMatchNet(config)

    def forward(self, p, option_len):
        oc_seq = p.new(p.size()).zero_()
        for i in range(p.size(0)):
            op = p[i]
            if i % 4 == 0:
                op1 = p[i + 1]
                op2 = p[i + 2]
                op3 = p[i + 3]
                indx = [i + 1, i + 2, i + 3]
            elif i % 4 == 1:
                op1 = p[i - 1]
                op2 = p[i + 1]
                op3 = p[i + 2]
                indx = [i - 1, i + 1, i + 2]
            elif i % 4 == 2:
                op1 = p[i - 2]
                op2 = p[i - 1]
                op3 = p[i + 1]
                indx = [i - 2, i - 1, i + 1]
            else:
                op1 = p[i - 3]
                op2 = p[i - 2]
                op3 = p[i - 1]
                indx = [i - 3, i - 2, i - 1]
            # oc1 = self.get_choice_interaction([op1.unsqueeze(0), op.unsqueeze(0), optlen.unsqueeze(0) + 1])
            # oc2 = self.get_choice_interaction([op2.unsqueeze(0), op.unsqueeze(0), optlen.unsqueeze(0) + 1])
            # oc3 = self.get_choice_interaction([op3.unsqueeze(0), op.unsqueeze(0), optlen.unsqueeze(0) + 1])
            oc1 = self.choice_match([op.unsqueeze(0), op1.unsqueeze(0), option_len[indx[0]].unsqueeze(0) + 1])
            oc2 = self.choice_match([op.unsqueeze(0), op2.unsqueeze(0), option_len[indx[1]].unsqueeze(0) + 1])
            oc3 = self.choice_match([op.unsqueeze(0), op3.unsqueeze(0), option_len[indx[2]].unsqueeze(0) + 1])
            cat_oc = torch.cat([oc1, oc2, oc3], 2)
            oc = self.dense3(cat_oc)
            oc_seq[i] = self.fusechoice([op, oc])
        return oc_seq


class DUMA(BertPreTrainedModel):
    def __init__(self, config, n_layers=1, n_head=12, dropout=0.1):
        super(DUMA, self).__init__(config)
        self.num_choices = 15
        self.bert = BertModel(config)
        d_model = config.hidden_size

        self.aoi = AOI(config)
        self.smatch = SingleMatchNet(config)
        self.fusenet = FuseNet(config)

        d_k, d_v = (d_model // n_head,) * 2
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.layers = nn.ModuleList([MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
                                     for _ in range(n_layers)])
        self.fuse3 = nn.Linear(d_model * 3, 2)
        # self.classifier = nn.Linear(config.hidden_size, 1)
        self.classifier_all = nn.Linear(4, self.num_choices)
        self.init_weights()

    def forward(self, input_ids, token_type_ids, attention_mask, doc_len, ques_len, option_len, single_prob, labels):
        flat_input_ids = input_ids.view(-1, input_ids.size(-1))  # [bs*4, sl]
        doc_len = doc_len.view(-1, doc_len.size(0) * doc_len.size(1)).squeeze()  # [1, bs*4]
        ques_len = ques_len.view(-1, ques_len.size(0) * ques_len.size(1)).squeeze()  # [1, bs*4]
        option_len = option_len.view(-1, option_len.size(0) * option_len.size(1)).squeeze()  # [1, bs*4]
        # print("input_ids:", input_ids.size())[1, 4, 512]
        flat_token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1))  # [bs*4, sl]
        flat_attention_mask = attention_mask.view(-1, attention_mask.size(-1))  # [bs*4, sl]
        sequence_output, pooled_output = self.bert(input_ids=flat_input_ids, token_type_ids=flat_token_type_ids, attention_mask=flat_attention_mask)[:2]

        doc_seq_output, ques_seq_output, option_seq_output, doc_seq_mask, ques_seq_mask, option_seq_mask = seperate_seq(sequence_output, doc_len, ques_len, option_len)
        option_seq_output_1 = self.aoi(option_seq_output, option_len)

        qa_output = self.smatch([ques_seq_output, option_seq_output_1, option_len + 1])
        aq_output = self.smatch([option_seq_output_1, ques_seq_output, ques_len + 1])
        qa_output_pool, _ = qa_output.max(1)
        aq_output_pool, _ = aq_output.max(1)
        qa_fuse = self.fusenet([qa_output_pool, aq_output_pool])

        hp, hq = ques_seq_output, option_seq_output
        pq_mask = get_mask(ques_seq_mask, option_seq_mask)
        qp_mask = get_mask(option_seq_mask, ques_seq_mask)

        for i, layer in enumerate(self.layers):
            hp, _ = layer(hp, hq, hq, mask=pq_mask)
            hq, _ = layer(hq, hp, hp, mask=qp_mask)

        hp, _ = hp.max(1)
        hq, _ = hq.max(1)
        cat_pq = torch.cat([hp, hq, qa_fuse], dim=1)
        output_pool = self.dropout(cat_pq)
        match_logits = self.fuse3(output_pool)
        match_reshaped_logits = match_logits.view(-1, 4)
        match_reshaped_logits = self.classifier_all(match_reshaped_logits)
        match_reshaped_logits_mask = torch.zeros_like(match_reshaped_logits, dtype=torch.bool)
        match_reshaped_logits_mask[single_prob == 1, 4:] = 1
        # match_reshaped_logits_clone.fill_(-1e4)
        match_reshaped_logits = match_reshaped_logits.masked_fill(mask=match_reshaped_logits_mask, value=torch.min(match_reshaped_logits))
        # if single_prob.nonzero().size(0):
        #     print(match_reshaped_logits_mask)
        #     print(single_prob)
        #     print(match_reshaped_logits)
        #     input()
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            match_loss = loss_fct(match_reshaped_logits, labels)
            return match_loss, match_reshaped_logits
        else:
            return match_reshaped_logits


class DUMA_2(BertPreTrainedModel):
    def __init__(self, config, n_layers=1, n_head=12, dropout=0.1):
        super(DUMA_2, self).__init__(config)
        self.num_choices = 15
        self.bert = BertModel(config)
        d_model = config.hidden_size

        self.aoi = AOI(config)
        self.smatch = SingleMatchNet(config)
        self.fusenet = FuseNet(config)

        d_k, d_v = (d_model // n_head,) * 2
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.layers = nn.ModuleList([MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
                                     for _ in range(n_layers)])
        self.fuse3 = nn.Linear(d_model * 3, 1)
        # self.classifier = nn.Linear(config.hidden_size, 1)
        self.classifier_all = nn.Linear(4, self.num_choices)
        self.init_weights()

    def forward(self, input_ids, token_type_ids, attention_mask, doc_len, ques_len, option_len, single_prob, labels):
        flat_input_ids = input_ids.view(-1, input_ids.size(-1))  # [bs*4, sl]
        doc_len = doc_len.view(-1, doc_len.size(0) * doc_len.size(1)).squeeze()  # [1, bs*4]
        ques_len = ques_len.view(-1, ques_len.size(0) * ques_len.size(1)).squeeze()  # [1, bs*4]
        option_len = option_len.view(-1, option_len.size(0) * option_len.size(1)).squeeze()  # [1, bs*4]
        # print("input_ids:", input_ids.size())[1, 4, 512]
        flat_token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1))  # [bs*4, sl]
        flat_attention_mask = attention_mask.view(-1, attention_mask.size(-1))  # [bs*4, sl]
        sequence_output, pooled_output = self.bert(input_ids=flat_input_ids, token_type_ids=flat_token_type_ids, attention_mask=flat_attention_mask)[:2]

        doc_seq_output, ques_seq_output, option_seq_output, doc_seq_mask, ques_seq_mask, option_seq_mask = seperate_seq(sequence_output, doc_len, ques_len, option_len)
        option_seq_output_1 = self.aoi(option_seq_output, option_len)

        qa_output = self.smatch([ques_seq_output, option_seq_output_1, option_len + 1])
        aq_output = self.smatch([option_seq_output_1, ques_seq_output, ques_len + 1])
        qa_output_pool, _ = qa_output.max(1)
        aq_output_pool, _ = aq_output.max(1)
        qa_fuse = self.fusenet([qa_output_pool, aq_output_pool])

        hp, hq = ques_seq_output, option_seq_output
        pq_mask = get_mask(ques_seq_mask, option_seq_mask)
        qp_mask = get_mask(option_seq_mask, ques_seq_mask)

        for i, layer in enumerate(self.layers):
            hp, _ = layer(hp, hq, hq, mask=pq_mask)
            hq, _ = layer(hq, hp, hp, mask=qp_mask)

        hp, _ = hp.max(1)
        hq, _ = hq.max(1)
        cat_pq = torch.cat([hp, hq, qa_fuse], dim=1)
        output_pool = self.dropout(cat_pq)
        match_logits = self.fuse3(output_pool)
        match_reshaped_logits = match_logits.view(-1, 4)
        match_reshaped_logits = self.classifier_all(match_reshaped_logits)
        match_reshaped_logits_mask = torch.zeros_like(match_reshaped_logits, dtype=torch.bool)
        match_reshaped_logits_mask[single_prob == 1, 4:] = 1
        # match_reshaped_logits_clone.fill_(-1e4)
        match_reshaped_logits = match_reshaped_logits.masked_fill(mask=match_reshaped_logits_mask, value=torch.min(match_reshaped_logits))
        # if single_prob.nonzero().size(0):
        #     print(match_reshaped_logits_mask)
        #     print(single_prob)
        #     print(match_reshaped_logits)
        #     input()
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            match_loss = loss_fct(match_reshaped_logits, labels)
            return match_loss, match_reshaped_logits
        else:
            return match_reshaped_logits


class DUMA_1(BertPreTrainedModel):
    def __init__(self, config, n_layers=1, d_model=768, n_head=12, dropout=0.1):
        super(DUMA_1, self).__init__(config)
        self.num_choices = 15
        self.bert = BertModel(config)
        d_k, d_v = (d_model // n_head,) * 2
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.layers = nn.ModuleList([nn.ModuleList([MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout),
                                                    MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)])
                                     for _ in range(n_layers)])
        self.fuse = nn.Linear(d_model * 2, 1)
        # self.classifier = nn.Linear(config.hidden_size, 1)
        self.classifier_all = nn.Linear(4, self.num_choices)
        self.init_weights()

    def forward(self, input_ids, token_type_ids, attention_mask, doc_len, ques_len, option_len, sentence_index, labels):
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
        # output_pool = self.dropout(cat_pq)
        match_logits = self.fuse(cat_pq)
        match_reshaped_logits = match_logits.view(-1, 4)
        match_reshaped_logits = self.classifier_all(match_reshaped_logits)
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            match_loss = loss_fct(match_reshaped_logits, labels)
            return match_loss, match_reshaped_logits
        else:
            return match_reshaped_logits
