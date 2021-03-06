from torch import nn
from transformers import BertPreTrainedModel, BertModel
import torch
from torch.autograd import Variable
import heapq
from torch.nn import CrossEntropyLoss


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


class DotMatchNet(nn.Module):
    def __init__(self, config):
        super(DotMatchNet, self).__init__()
        self.map_linear = nn.Linear(450, config.hidden_size)
        self.trans_linear = nn.Linear(config.hidden_size, config.hidden_size)

    def forward(self, inputs):
        proj_p, proj_q, seq_len = inputs
        trans_q = self.trans_linear(proj_q)
        # trans_p = self.trans_linear(proj_p)
        att_weights = proj_p.bmm(torch.transpose(proj_q, 1, 2))
        output = nn.ReLU()(self.map_linear(att_weights))
        return output


# class SingleMatchNet(nn.Module):
#     def __init__(self, config):
#         super(SingleMatchNet, self).__init__()
#         self.map_linear = nn.Linear(2 * config.hidden_size, 2 * config.hidden_size)
#         self.trans_linear = nn.Linear(config.hidden_size, config.hidden_size)
#
#     def forward(self, inputs):
#         proj_p, proj_q, seq_len = inputs
#         trans_q = self.trans_linear(proj_q)
#         att_weights = proj_p.bmm(torch.transpose(trans_q, 1, 2))
#         att_norm = masked_softmax(att_weights, seq_len)
#         att_vec = att_norm.bmm(proj_q)
#         output = nn.ReLU()(self.trans_linear(att_vec))
#         return output


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


class AOI(nn.Module):
    def __init__(self, config):
        super(AOI, self).__init__()
        self.trans_linear = nn.Linear(config.hidden_size, config.hidden_size)
        self.dense3 = nn.Linear(3 * config.hidden_size, config.hidden_size)
        self.fusechoice = FuseNet(config)
        self.choice_match = SingleMatchNet(config)

    # def get_choice_interaction(self, inputs):
    #     proj_p, proj_q, seq_len = inputs
    #     trans_q = self.trans_linear(proj_q)
    #     att_weights = proj_p.bmm(torch.transpose(trans_q, 1, 2))
    #     att_norm = masked_softmax(att_weights, seq_len)
    #     att_vec = att_norm.bmm(proj_q)
    #     output = nn.ReLU()(att_vec)
    #     return output

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


class SoftSel(nn.Module):
    def __init__(self, config):
        super(SoftSel, self).__init__()
        self.map_linear = nn.Linear(2 * config.hidden_size, 2 * config.hidden_size)
        self.trans_linear = nn.Linear(config.hidden_size, config.hidden_size)

    def forward(self, inputs):
        proj_p, proj_q, seq_len = inputs
        trans_q = self.trans_linear(proj_q)
        att_weights = proj_p.bmm(torch.transpose(trans_q, 1, 2))
        att_norm = masked_softmax(att_weights, seq_len)
        att_vec = att_norm.bmm(proj_q)
        output = nn.ReLU()(self.trans_linear(att_vec))
        return output


class Match(nn.Module):
    def __init__(self, config):
        super(Match, self).__init__()
        self.map_linear = nn.Linear(2 * config.hidden_size, 2 * config.hidden_size)
        self.trans_linear = nn.Linear(config.hidden_size, config.hidden_size)
        self.p_linear = nn.Linear(config.hidden_size, 1)
        self.linear21 = nn.Linear(2 * config.hidden_size, config.hidden_size)
        # self.bigru = nn.GRU(config.hidden_size, config.hidden_size, bidirectional=True)

    def forward(self, inputs):
        p1, p, q1, q, seq_len = inputs
        con1 = torch.cat([p1 - p, p1 * p], 2)
        con2 = torch.cat([q1 - q, q1 * q], 2)
        m1 = nn.ReLU()(self.linear21(con1))
        m2 = nn.ReLU()(self.linear21(con2))
        hm = self.linear21(torch.cat([m1, m2], 2))
        # hm = self.bigru(m)

        hmax, _ = hm.max(1)
        # proj_linear = nn.Linear(hm.size(1), 1)
        # alpha = masked_softmax(nn.ReLU()(self.p_linear(hm)), seq_len)
        # hatt = torch.transpose(hm, 1, 2).bmm(alpha).squeeze()
        # output = torch.cat([hmax, hatt], 1)
        return hmax


class DualMatchNet(nn.Module):
    def __init__(self, config):
        super(DualMatchNet, self).__init__()
        self.map_linear = nn.Linear(2 * config.hidden_size, 2 * config.hidden_size)
        self.trans_linear = nn.Linear(config.hidden_size, config.hidden_size)
        self.linear4 = nn.Linear(4 * config.hidden_size, 4 * config.hidden_size)
        self.linear3 = nn.Linear(3 * config.hidden_size, 3 * config.hidden_size)
        self.linear2 = nn.Linear(2 * config.hidden_size, 2 * config.hidden_size)
        self.linear = nn.Linear(config.hidden_size, config.hidden_size)
        self.drop_module = nn.Dropout(2 * config.hidden_dropout_prob)
        self.fuse = FuseNet(config)
        self.rank_module = nn.Linear(config.hidden_size * 2, 1)

    def forward(self, inputs):
        proj_p, proj_q, len_q = inputs
        # print(len_q)
        trans_q = self.trans_linear(proj_q)
        trans_p = self.trans_linear(proj_p)
        att_weights = proj_p.bmm(torch.transpose(trans_q, 1, 2))
        # print("attweight: ", att_weights.size())
        att_norm = masked_softmax(att_weights, len_q)
        # print("att_norm: ", att_norm.size())

        att_doc = att_norm.bmm(proj_q)
        doc_pool, _ = att_doc.max(1)

        att_ques = torch.transpose(att_norm, 1, 2).bmm(trans_p)
        ques_pool, _ = att_ques.max(1)
        # elem_min = att_vec - proj_p
        # elem_mul = att_vec * proj_p
        # con1 = torch.cat([elem_min, elem_mul], 2)
        # con2 = torch.cat([att_vec, att_vec], 2)
        fuse_vec = self.fuse([doc_pool, ques_pool])
        output = nn.ReLU()(self.linear(fuse_vec))
        return output


class MatchNet(nn.Module):
    def __init__(self, config):
        super(MatchNet, self).__init__()
        self.map_linear = nn.Linear(2 * config.hidden_size, 2 * config.hidden_size)
        self.trans_linear = nn.Linear(config.hidden_size, config.hidden_size)

    def forward(self, inputs):
        proj_p, proj_q, seq_len = inputs
        trans_q = self.trans_linear(proj_q)
        att_weights = proj_p.bmm(torch.transpose(trans_q, 1, 2))
        # print("attweight: ", att_weights.size())
        att_norm = masked_softmax(att_weights, seq_len)
        # print("att_norm: ", att_norm.size())

        att_vec = att_norm.bmm(proj_q)
        elem_min = att_vec - proj_p
        elem_mul = att_vec * proj_p
        all_con = torch.cat([elem_min, elem_mul], 2)
        output = nn.ReLU()(self.map_linear(all_con))
        return output


#
# def seperate_seq(sequence_output, doc_len, ques_len, option_len):
#     doc_seq_output = sequence_output.new(sequence_output.size()).zero_()
#     doc_ques_seq_output = sequence_output.new(sequence_output.size()).zero_()
#     ques_seq_output = sequence_output.new(sequence_output.size()).zero_()
#     ques_option_seq_output = sequence_output.new(sequence_output.size()).zero_()
#     option_seq_output = sequence_output.new(sequence_output.size()).zero_()
#     for i in range(doc_len.size(0)):
#         doc_seq_output[i, :doc_len[i]] = sequence_output[i, :doc_len[i]]
#         doc_ques_seq_output[i, :doc_len[i] + ques_len[i]] = sequence_output[i, :doc_len[i] + ques_len[i]]
#         ques_seq_output[i, :ques_len[i]] = sequence_output[i, doc_len[i] + 1:doc_len[i] + ques_len[i] + 1]
#         ques_option_seq_output[i, :ques_len[i]+option_len[i]] = sequence_output[i, doc_len[i] + 1:doc_len[i] + ques_len[i] + option_len[i] + 1]
#         option_seq_output[i, :option_len[i]] = sequence_output[i,
#                                                  doc_len[i] + ques_len[i] + 1:doc_len[i] + ques_len[i] + option_len[
#                                                    i] + 1]
#     return doc_ques_seq_output, ques_option_seq_output, doc_seq_output, ques_seq_output, option_seq_output

# def seperate_seq(sequence_output, doc_len, ques_len, option_len):
#     doc_seq_output = sequence_output.new(sequence_output.size()).zero_()
#     doc_ques_seq_output = sequence_output.new(sequence_output.size()).zero_()
#     ques_seq_output = sequence_output.new(sequence_output.size()).zero_()
#     ques_option_seq_output = sequence_output.new(sequence_output.size()).zero_()
#     option_seq_output = sequence_output.new(sequence_output.size()).zero_()
#     for i in range(doc_len.size(0)):
#         doc_seq_output[i, :doc_len[i]] = sequence_output[i, 1:doc_len[i] + 1]
#         doc_ques_seq_output[i, :doc_len[i] + ques_len[i]] = sequence_output[i, :doc_len[i] + ques_len[i]]
#         ques_seq_output[i, :ques_len[i]] = sequence_output[i, doc_len[i] + 2:doc_len[i] + ques_len[i] + 2]
#         ques_option_seq_output[i, :ques_len[i] + option_len[i]] = sequence_output[i, doc_len[i] + 1:doc_len[i] + ques_len[i] + option_len[i] + 1]
#         option_seq_output[i, :option_len[i]] = sequence_output[i,
#                                                doc_len[i] + ques_len[i] + 2:doc_len[i] + ques_len[i] + option_len[
#                                                    i] + 2]
#     return doc_ques_seq_output, ques_option_seq_output, doc_seq_output, ques_seq_output, option_seq_output


def seperate_seq(sequence_output, doc_len, ques_len, option_len):
    doc_seq_output = sequence_output.new(sequence_output.size()).zero_()
    # doc_ques_seq_output = sequence_output.new(sequence_output.size()).zero_()
    ques_seq_output = sequence_output.new(sequence_output.size()).zero_()
    # ques_option_seq_output = sequence_output.new(sequence_output.size()).zero_()
    option_seq_output = sequence_output.new(sequence_output.size()).zero_()
    for i in range(doc_len.size(0)):
        doc_seq_output[i, :doc_len[i]] = sequence_output[i, 1:doc_len[i] + 1]
        # doc_ques_seq_output[i, :doc_len[i] + ques_len[i]] = sequence_output[i, 1:doc_len[i] + ques_len[i] + 1]
        ques_seq_output[i, :ques_len[i]] = sequence_output[i, doc_len[i] + 1:doc_len[i] + ques_len[i] + 1]
        # ques_option_seq_output[i, :ques_len[i] + option_len[i] + 1] = sequence_output[i, doc_len[i] + 1:doc_len[i] + ques_len[i] + option_len[i] + 2]
        option_seq_output[i, :option_len[i]] = sequence_output[i, doc_len[i] + ques_len[i] + 2:doc_len[i] + ques_len[i] + option_len[i] + 2]
    return doc_seq_output, ques_seq_output, option_seq_output


def cos_distance(s1, s2):
    a = s1
    b = s2

    ca = a.repeat(b.size(0), 1)
    cca = ca.view(b.size(0), a.size(0), -1)

    cb = b.repeat(1, a.size(0))
    ccb = cb.view(b.size(0), a.size(0), -1)
    # print(cca.size(), ccb.size())
    cos = torch.nn.CosineSimilarity(2)
    res = cos(cca, ccb)
    return res


def get_score(sentence, question, option):  # sentence: sentence_len*768 question: question_len*768
    # result_sent_ques = sentence.new(sentence.size(0), question.size(0)).zero_()
    # result_sent_option = sentence.new(sentence.size(0), option.size(0)).zero_()

    result_sent_ques = cos_distance(sentence, question)  # question.size(0)* sentence.size(0)
    result_sent_option = cos_distance(sentence, option)  # option.size(0)* sentence.size(0)

    res_ques, _ = result_sent_ques.max(1)
    res_option, _ = result_sent_option.max(1)

    # sent1, _ = result_sent_ques.max(0)
    # sent2, _ = result_sent_option.max(0)
    # print("pool1, pool2:", res_ques,sent1)

    # result = np.sum(heapq.nlargest(2,res_ques)) + np.sum(heapq.nlargest(2,res_option))
    result = torch.sum(res_ques) / res_ques.size(0) + torch.sum(res_option) / res_option.size(0)
    # result = torch.sum(sent1) / sent1.size(0) + torch.sum(sent2) / sent2.size(0)
    # print("res_ques, res_option:", torch.sum(res_ques), torch.sum(res_option)/res_option.size(0))
    return result


class BertForMultipleChoiceWithMatch(BertPreTrainedModel):

    def __init__(self, config, model_choices=300):
        super(BertForMultipleChoiceWithMatch, self).__init__(config)
        self.num_choices = 4
        self.model_choices = model_choices
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(1 * config.hidden_size, 1)
        self.init_weights()

    def forward(self, input_ids=None, token_type_ids=None, attention_mask=None, doc_len=None, ques_len=None, option_len=None, sentence_index=None, labels=None, is_3=False):
        flat_input_ids = input_ids.view(-1, input_ids.size(-1))  # [bs*4, sl]
        flat_token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1))  # [bs*4, sl]
        flat_attention_mask = attention_mask.view(-1, attention_mask.size(-1))  # [bs*4, sl]
        # with torch.no_grad():
        sequence_output, pooled_output = self.bert(input_ids=flat_input_ids, token_type_ids=flat_token_type_ids, attention_mask=flat_attention_mask)[:2]

        pooled_output = self.dropout(pooled_output)
        match_logits = self.classifier(pooled_output)
        match_reshaped_logits = match_logits.view(-1, self.num_choices)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            match_loss = loss_fct(match_reshaped_logits, labels)
            # high_match_loss = loss_fct(high_match_reshaped_logits, labels)
            return match_loss, match_reshaped_logits
        else:
            return match_reshaped_logits
