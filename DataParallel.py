import torch
import torch.nn.functional as F


class DataParallel(torch.nn.DataParallel):
    def __init__(self, *init, input_pad=6, output_left_cut=2, output_right_cut=2):
        super(DataParallel, self).__init__(*init)
        self.input_pad = input_pad
        self.output_left_cut = output_left_cut
        self.output_right_cut = output_right_cut

    def forward(self, *inputs):
        # 0, 1, 2, 3
        if len(self.device_ids) == 1:
            return ((self.module(*inputs[0]), ), )

        n = len(inputs)
        temp = [[[]] for _ in range(n)]
        for i in range(n - 1):
            if inputs[i][0].size(3) >= self.input_pad:
                temp[i + 1][0].append(inputs[i][0][:, :, :, -self.input_pad:].cuda(self.device_ids[i + 1]))
            else:
                this_input = F.pad(inputs[i][0], (self.input_pad - inputs[i][0].size(3), 0, 0, 0))
                temp[i + 1][0].append(this_input[:, :, :, -self.input_pad:].cuda(self.device_ids[i + 1]))
        for i in range(n):
            temp[i][0].append(inputs[i][0])
        for i in range(n - 1):
            if inputs[i][0].size(3) >= self.input_pad:
                temp[i][0].append(inputs[i + 1][0][:, :, :, :self.input_pad].cuda(self.device_ids[i]))
            else:
                this_input = F.pad(inputs[i + 1][0], (0, self.input_pad - inputs[i + 1][0].size(3), 0, 0))
                temp[i][0].append(this_input[:, :, :, :self.input_pad].cuda(self.device_ids[i]))
        for i in range(n):
            temp[i] = (torch.cat(temp[i][0], dim=3), )
        temp = tuple(temp)

        replicas = self.replicate(self.module, self.device_ids[:len(inputs)])
        outputs = self.parallel_apply(replicas, temp, ({}, ) * len(inputs))

        for i in range(n):
            if i == 0:
                outputs[i] = outputs[i][:, :, :, :-self.output_right_cut]
            elif i == n - 1:
                outputs[i] = outputs[i][:, :, :, self.output_left_cut:]
            else:
                outputs[i] = outputs[i][:, :, :, self.output_left_cut:-self.output_right_cut]
            if not isinstance(outputs[i], tuple):
                outputs[i] = (outputs[i], )
        return tuple(outputs)
