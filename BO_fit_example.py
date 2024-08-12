# !/usr/bin/env python
# coding: utf-8

import os, shutil, pickle, time, traceback
import numpy as np
import pandas as pd
import torch

from src.bayesian_optimization import ParamsFitting, PFIO, PFExecuteModule

os.system('export OMP_NUM_THREADS=24')
os.system('ulimit -v 48000000')

root_path = os.getcwd()+'/'

class IO(PFIO):
    
    def __init__(self, root_path, input_file_name='in.Quanty', output_file_name='out.Quanty', input_templates_file_name='10_RIXS_L23_M45.lua', target_file_name='Literature_value.Quanty'):
        self.root_path = root_path
        self.input_file_name = input_file_name
        self.output_file_name = output_file_name
        self.input_templates_file_name = input_templates_file_name
        self.target_file_name = target_file_name

    def read_output(self, folder):

        with open(f'{self.root_path}{folder}/{self.output_file_name}', 'r') as file:
            lines = file.readlines()

        data = []
        for i, line in enumerate(lines):
            if 5 <= i:
                values = line.split()
                data.append([float(values[j]) for j in range(2, len(values), 2)])

        # Convert to numpy array
        data_array = -np.array(data)
        data_mat = data_array/max(np.max(data_array), 0.0001)

        return data_mat

    def read_target(self, folder):

        with open(f'{self.root_path}{folder}/{self.target_file_name}', 'r') as file:
            lines = file.readlines()

        data = []
        for i, line in enumerate(lines):
            if 5 <= i:
                values = line.split()
                data.append([float(values[j]) for j in range(2, len(values), 2)])

        # Convert to numpy array
        data_array = -np.array(data)
        data_mat = data_array/max(np.max(data_array), 0.0001)

        return data_mat
            
    ###### modify the input ######
    def modify_input(self, dict_para):
        with open(self.root_path+'files/'+self.input_templates_file_name,'r') as inf:
            inl = inf.readlines()

        for i in range(len(inl)):
            if "--input parameter (variable)" in inl[i]:
                linfo = inl[i].split()
                para_name = linfo[0]
                ninfo = linfo[0]+' = '+"{:.18f}".format(dict_para[linfo[0]])+"  --input parameter (variable)\n"
                inl[i] = ninfo

        with open(self.root_path+'/'+self.input_file_name, 'w') as outf:
            outf.writelines(inl)


class Execute_module(PFExecuteModule):
    def __init__(self, dim=15, lb=None, ub=None):
        self.dim = dim
        self.lb = lb
        self.ub = ub
        self.bounds = np.stack([self.lb, self.ub])
                
    def __call__(self, X):
        dimension = X.ndim
        if X.ndim == 1:
            X = np.expand_dims(X, axis=0)
        assert len(X[0]) == self.dim
        assert X.ndim == 2
        # assert np.all(X <= self.ub) and np.all(X >= self.lb)
        X = np.clip(X, 0, np.inf)
        X = torch.tensor(X)
        
        os.chdir(root_path)
        io = IO(root_path)
        
        ### loss function
        #loss_func = torch.nn.SmoothL1Loss(reduction='sum')
        # loss_func = torch.nn.SmoothL1Loss(reduction='mean')
        #loss_func = torch.nn.L1Loss(reduction='mean')
        #loss_funcsum = torch.nn.L1Loss(reduction='sum')
        loss_func = torch.nn.MSELoss(reduction='mean')

        ### 1. read in target_file, get target value;
        ### --------------------------------------------------------------------------------- ###
        LY = io.read_target(folder='files', )
        base_mat = LY / max(np.max(LY), 0.01)
        ### --------------------------------------------------------------------------------- ###


        ### 2. modify the input file/files according to X/Xs
        ### --------------------------------------------------------------------------------- ###
        folder_init = int(os.popen(f'cat {root_path}/Data/fit_res.csv | wc -l').read())
        for folder_num, x in enumerate(X):
            
            folder = str(folder_num+folder_init)
            if not os.path.isdir(root_path+'previous/'):
                os.mkdir(root_path+'previous/')
            if not os.path.isdir(root_path+'previous/'+folder):
                os.mkdir(root_path+'previous/'+folder)
            
            Num_x = [i.item() for i in x]
            
            quanty_dict = {'Udd':Num_x[0], 'Upd':Num_x[1], 'Delta':Num_x[2], 'F2dd':Num_x[3], 'F4dd':Num_x[4], 'F2pd':Num_x[5],
                           'G1pd':Num_x[6], 'G3pd':Num_x[7], 'tenDq':Num_x[8], 'tenDqL':Num_x[9], 'Veg':Num_x[10], 
                           'Vt2g':Num_x[11], 'zeta_3d':Num_x[12], 'zeta_2p':Num_x[13], 'H112':Num_x[14]}
            
            io.modify_input(quanty_dict)
            src_in_file = os.path.join(root_path, 'in.Quanty')
            shutil.move(src_in_file, root_path+'previous/'+folder+'/in.Quanty')
            os.chdir(root_path+'previous/'+folder)
            os.popen('OMP_NUM_THREADS=8 nohup /home/phD/yixuan/softwares/Quanty/Quanty in.Quanty  >/dev/null 2>&1 &').read()  ### submit all simulation tasks
            os.chdir(root_path)
        ### --------------------------------------------------------------------------------- ###
            

        ### 3. Executing simulation/experiment to get output file
        ### --------------------------------------------------------------------------------- ###
        ### wait until all jobs finished
        cal_len = int(os.popen("ps -aux | grep '[Q]uanty' | wc -l").read())
        while cal_len!=0:
            cal_len = int(os.popen("ps -aux | grep '[Q]uanty' | wc -l").read())
            time.sleep(1)
        ### --------------------------------------------------------------------------------- ###

        
        ### 4. compare loss between output and target: eg. loss = abs(target-output)
        ### --------------------------------------------------------------------------------- ###
        Y = []
        ### Collect results
        for folder_num, x in enumerate(X):

            folder = str(folder_num+folder_init)
            Num_x = [i.item() for i in x]
            
            judge_num = 0
            while not os.path.isfile(root_path+'previous/'+folder+'/out.Quanty'):
                print('no result calculated, retry')
                os.chdir(root_path+'previous/'+folder)
                os.popen('OMP_NUM_THREADS=8 nohup /home/phD/yixuan/softwares/Quanty/Quanty in.Quanty  >/dev/null 2>&1 &').read()
                judge_num += 1
                cal_len = int(os.popen("ps -aux | grep '[Q]uanty' | wc -l").read())
                while cal_len!=0:
                    cal_len = int(os.popen("ps -aux | grep '[Q]uanty' | wc -l").read())
                    print('wait 1s')
                    time.sleep(1)
                if judge_num >= 2:
                    print('Calculation failed, parameter not possible')
                    shutil.copy(root_path+'previous/0/out.Quanty', root_path+'previous/'+folder+'/out.Quanty')
            os.chdir(root_path)

            _res = io.read_output('previous/'+folder)
            fbase_mat = _res / max(np.max(_res), 0.0001)

            target = np.clip(base_mat*10000, 0, 1)
            fitted = np.clip(fbase_mat*10000, 0, 1)

            loss = (
                    loss_func(torch.tensor(fitted), torch.tensor(target))
                   ).detach().cpu().numpy()
            
            Y.append([loss])
        ### --------------------------------------------------------------------------------- ###
            
        ### 5. write to csv file by following command:  self.append_to_csv(root_path, X, loss)
        ### --------------------------------------------------------------------------------- ###
        output = np.array(Y)
        self.append_to_csv(root_path, X.detach().cpu().numpy(), output)
        ### --------------------------------------------------------------------------------- ###

        ### 6. return loss
        ### --------------------------------------------------------------------------------- ###
        return output
        ### --------------------------------------------------------------------------------- ###



###### set the bound limitation for bayesian ######

lb = np.array([3,  4,  2, 5,  4,  3,  2, 1, 0, 0, 0, 0, 0, 5,  0])
ub = np.array([11, 14, 8, 15, 12, 10, 8, 4, 2, 3, 4, 4, 1, 15, 1])

fun_q = Execute_module(dim=len(lb), lb=lb, ub=ub)

### Run the code
os.chdir(root_path)
io = IO(root_path)

method = 'BO_Boosting'
if method == 'BWO':
    fitting_results = ParamsFitting(fun_q).BO_BWO()
elif method == 'BO_Turbo':
    fitting_results = ParamsFitting(fun_q).BO_Turbo()
elif method == 'BO_Turbo_bwo':
    fitting_results = ParamsFitting(fun_q).BO_Turbo_bwo()
else:
    fitting_results = ParamsFitting(fun_q).BO_Boosting()

fun_q.append_to_csv(root_path, fitting_results['X_best'], fitting_results['Best_value'], file_name='final_res.csv')

# save results
# csv_file = os.path.join(root_path, 'final_res.csv')
# data = {'index' + str(i): [fitting_results['X_best'][i]] for i in range(len(fitting_results['X_best']))}
# data['loss'] = [fitting_results['Best_value']]
# df = pd.DataFrame(data)

# if not os.path.isfile(csv_file):
#     df.to_csv(csv_file, index=False)
# else:
#     df.to_csv(csv_file, mode='a', header=False, index=False)


