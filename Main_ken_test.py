"""
Implementation of "Error Correction Code Transformer" (ECCT)
https://arxiv.org/abs/2203.14966
@author: Yoni Choukroun, choukroun.yoni@gmail.com
"""
from __future__ import print_function
import argparse
import random
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
from torch.utils.data import DataLoader
from torch.utils import data
from datetime import datetime
import logging
from Codes import *
import time
from torch.optim.lr_scheduler import CosineAnnealingLR
from Model import ECC_Transformer
import math
from Distance_implementations import *
from Distance_implementations import gaussian_kernel
from Distance_implementations import polynomial_kernel
from Distance_implementations import compute_mmd
from Distance_implementations import frechet_distance_numpy
from Distance_implementations import matrix_sqrt
from Distance_implementations import llr
from Distance_implementations import belief_propagation
import numpy as np
import matplotlib.pyplot as plt




##################################################################
##################################################################

def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

##################################################################


class ECC_Dataset(data.Dataset):
    def __init__(self, code, sigma, len, zero_cw=True):
        self.code = code
        self.sigma = sigma
        self.len = len
        self.generator_matrix = code.generator_matrix.transpose(0, 1)
        self.pc_matrix = code.pc_matrix.transpose(0, 1)

        self.zero_word = torch.zeros((self.code.k)).long() if zero_cw else None
        self.zero_cw = torch.zeros((self.code.n)).long() if zero_cw else None

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        if self.zero_cw is None:
            m = torch.randint(0, 2, (1, self.code.k)).squeeze()
            x = torch.matmul(m, self.generator_matrix) % 2
        else:
            m = self.zero_word
            x = self.zero_cw
        z = torch.randn(self.code.n) * random.choice(self.sigma)
        y = bin_to_sign(x) + z
        magnitude = torch.abs(y)
        syndrome = torch.matmul(sign_to_bin(torch.sign(y)).long(),
                                self.pc_matrix) % 2
        syndrome = bin_to_sign(syndrome)
        return m.float(), x.float(), z.float(), y.float(), magnitude.float(), syndrome.float()


##################################################################
##################################################################

def train(model, device, train_loader, optimizer, epoch, LR):
    model.train()
    cum_loss = cum_ber = cum_fer = cum_samples = 0
    t = time.time()
    for batch_idx, (m, x, z, y, magnitude, syndrome) in enumerate(
            train_loader):
        z_mul = (y * bin_to_sign(x))
        z_pred = model(magnitude.to(device), syndrome.to(device))
        loss, x_pred = model.loss(-z_pred, z_mul.to(device), y.to(device))
        model.zero_grad()
        loss.backward()
        optimizer.step()
        ###
        ber = BER(x_pred, x.to(device))
        fer = FER(x_pred, x.to(device))

        cum_loss += loss.item() * x.shape[0]
        cum_ber += ber * x.shape[0]
        cum_fer += fer * x.shape[0]
        cum_samples += x.shape[0]
        if (batch_idx+1) % 500 == 0 or batch_idx == len(train_loader) - 1:
            logging.info(
                f'Training epoch {epoch}, Batch {batch_idx + 1}/{len(train_loader)}: LR={LR:.2e}, Loss={cum_loss / cum_samples:.2e} BER={cum_ber / cum_samples:.2e} FER={cum_fer / cum_samples:.2e}')
    logging.info(f'Epoch {epoch} Train Time {time.time() - t}s\n')
    return cum_loss / cum_samples, cum_ber / cum_samples, cum_fer / cum_samples


##################################################################

def test(model, device, test_loader_list, EbNo_range_test, min_FER=100,test_size_number=100):
   
    model.eval()
    test_loss_list, test_loss_ber_list, test_loss_fer_list, cum_samples_all = [], [], [], []
    t = time.time()
    out1=[]
    out2=[]
    out3=[]

    
    with torch.no_grad():
        for ii, test_loader in enumerate(test_loader_list):
            test_loss = test_ber = test_fer = cum_count = 0.
            ok=0
            print(f'ii={ii}')
           
            while True:
                (m, x, z, y, magnitude, syndrome) = next(iter(test_loader))
                ok=ok+1
                print(ok)
                z_mul = (y * bin_to_sign(x))
                z_pred = model(magnitude.to(device), syndrome.to(device))
                loss, x_pred = model.loss(-z_pred, z_mul.to(device), y.to(device))
                print(f'sum_x={torch.sum(x)}')
                # print(f'x_pred.shape={x_pred.shape}')
                # print(f'y.shape={y.shape}')
               
                # if ii==0:
                    
                #     data_test_save_x_1[cnt,:]=x.numpy()
                #     data_test_save_y_1[cnt,:]=y.numpy()
                #     data_test_save_x_hat_1[cnt,:]=x_pred.numpy()
                    
                # if ii==1:
                     
                #      data_test_save_x_2[cnt,:]=x.numpy()
                #      data_test_save_y_2[cnt,:]=y.numpy()
                #      data_test_save_x_hat_2[cnt,:]=x_pred.numpy()
                     
                # if ii==2:
                         
                #          data_test_save_x_3[cnt,:]=x.numpy()
                #          data_test_save_y_3[cnt,:]=y.numpy()
                #          data_test_save_x_hat_3[cnt,:]=x_pred.numpy()
              

                test_loss += loss.item() * x.shape[0] 

                test_ber += BER(x_pred, x.to(device)) * x.shape[0]
                test_fer += FER(x_pred, x.to(device)) * x.shape[0]
                cum_count += x.shape[0]
                ham=torch.sum(x != x_pred).item()
                #print(f'Hamming Distance;={ham}')
                if ii==0:
                      out1.append({ 'x_bin':x,
                     'recived': y,
                     
                     'Decode':x_pred,
                     'Haming_distance':ham})
                      
                elif ii==1:
                    out2.append({ 'x_bin':x,
                   'recived': y,
                   
                   'Decode':x_pred,
                   'Haming_distance':ham})
                else:
                    out3.append({ 'x_bin':x,
                   'recived': y,
                   
                   'Decode':x_pred,
                   'Haming_distance':ham})
                if cum_count > test_size_number-1:
                    #print(f'gookkkk={x.shape}')
                    break
                '''
                if (min_FER > 0 and test_fer > min_FER and cum_count > 1e2) or cum_count >= 1e2:#1e9
                    if cum_count >= 1e9:
                        print(f'Number of samples threshold reached for EbN0:{EbNo_range_test[ii]}')
                    else:    
                        print(f'FER count threshold reached for EbN0:{EbNo_range_test[ii]}')
                    break
              '''
            
              
     
            cum_samples_all.append(cum_count)
            test_loss_list.append(test_loss / cum_count)
            test_loss_ber_list.append(test_ber / cum_count)
            test_loss_fer_list.append(test_fer / cum_count)
            #print(f'Test EbN0={EbNo_range_test[ii]}, BER={test_loss_ber_list[-1]:.2e}')
        ###
        logging.info('\nTest Loss ' + ' '.join(
            ['{}: {:.2e}'.format(ebno, elem) for (elem, ebno)
             in
             (zip(test_loss_list, EbNo_range_test))]))
        logging.info('Test FER ' + ' '.join(
            ['{}: {:.2e}'.format(ebno, elem) for (elem, ebno)
             in
             (zip(test_loss_fer_list, EbNo_range_test))]))
        logging.info('Test BER ' + ' '.join(
            ['{}: {:.2e}'.format(ebno, elem) for (elem, ebno)
             in
             (zip(test_loss_ber_list, EbNo_range_test))]))
        logging.info('Test -ln(BER) ' + ' '.join(
            ['{}: {:.2e}'.format(ebno, -np.log(elem)) for (elem, ebno)
             in
             (zip(test_loss_ber_list, EbNo_range_test))]))
    logging.info(f'# of testing samples: {cum_samples_all}\n Test Time {time.time() - t} s\n')
    out=[out1,out2,out3]
    return out, [test_loss_list, test_loss_ber_list, test_loss_fer_list]

##################################################################
##################################################################
##################################################################


def main(args):
    code = args.code
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'device={device}')
    #################################
    #model = ECC_Transformer(args, dropout=0).to(device)
    model=torch.load('D:/python-codes/python-new/jalali-LDPC/ken_asking/codes_for_test_train/Test_ECCT/best_model_LDPC_ECCT_n_121_k_80_August_2_epochs_1000.pt',map_location=torch.device('cpu'), weights_only=False)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    logging.info(model)
    logging.info(f'# of Parameters: {np.sum([np.prod(p.shape) for p in model.parameters()])}')
    #################################
    EbNo_range_test = args.EbNo_range_test
    # EbNo_range_train = range(2, 8)
    # std_train = [EbN0_to_std(ii, code.k / code.n) for ii in EbNo_range_train]
    std_test = [EbN0_to_std(ii, code.k / code.n) for ii in EbNo_range_test]
    # train_dataloader = DataLoader(ECC_Dataset(code, std_train, len=args.batch_size * 1000, zero_cw=True), batch_size=int(args.batch_size),
    #                               shuffle=True, num_workers=args.workers)
    test_dataloader_list = [DataLoader(ECC_Dataset(code, [std_test[ii]], len=int(args.test_batch_size), zero_cw=False),
                                       batch_size=int(args.test_batch_size), shuffle=False, num_workers=args.workers) for ii in range(len(std_test))]
    #################################
    # best_loss = float('inf')
    # for epoch in range(1, args.epochs + 1):
    #     loss, ber, fer = train(model, device, train_dataloader, optimizer,
    #                            epoch, LR=scheduler.get_last_lr()[0])
    #     scheduler.step()
    #     if loss < best_loss:
    #         best_loss = loss
    #         torch.save(model, os.path.join(args.path, 'best_model'))
        # if epoch % 300 == 0 or epoch in [1, args.epochs]:
    time1_test=time.time()        
    output_test_nueral,test_dataset_nural=test(model, device, test_dataloader_list, EbNo_range_test,test_size_number=args.test_records)
    time2_test=time.time()
    time_duration_test=time2_test-time1_test
    print(f'time_duration_test={time_duration_test}')
    return output_test_nueral,test_dataset_nural


########################################################
##Classical Decoder
class SumProductDecoder:
    def __init__(self, H, max_iter=50):
        """
        Initialize LDPC decoder with parity-check matrix H
        
        Args:
            H: Parity-check matrix (m x n)
            max_iter: Maximum decoding iterations
        """
        self.H = H.astype(int)
        self.m, self.n = H.shape  # m checks, n bits
        self.max_iter = max_iter
        
        # Precompute connections
        self.var_nodes = [np.where(H[:, j] == 1)[0] for j in range(self.n)]
        self.check_nodes = [np.where(H[i, :] == 1)[0] for i in range(self.m)]
        
    def decode(self, y, sigma, early_term=True):
        """
        Decode received signal using Sum-Product Algorithm
        
        Args:
            y: Received signal (n bits)
            sigma: Noise standard deviation
            early_term: Stop if parity checks are satisfied
            
        Returns:
            x_hat: Decoded bits (0/1)
            n_iter: Number of iterations used
        """
        # Initialization
        Lc = 2 * y / sigma**2  # Channel LLRs
        
        # Variable to check messages (q_ij)
        Q = np.zeros((self.m, self.n))
        for i in range(self.m):
            for j in self.check_nodes[i]:
                Q[i, j] = Lc[j]
        
        # Main iteration loop
        for n_iter in range(1, self.max_iter + 1):
            # Check to variable messages (r_ij)
            R = np.zeros((self.m, self.n))
            for i in range(self.m):
                for j in self.check_nodes[i]:
                    # Product of tanh(q_ij'/2) for all j' ≠ j
                    prod = 1.0
                    for j_prime in self.check_nodes[i]:
                        if j_prime != j:
                            prod *= np.tanh(Q[i, j_prime] / 2)
                    R[i, j] = 2 * np.arctanh(prod)
            
            # Variable to check messages (q_ij)
            Q_new = np.zeros((self.m, self.n))
            for j in range(self.n):
                for i in self.var_nodes[j]:
                    # Sum of r_i'j + Lc_j for all i' ≠ i
                    sum_r = Lc[j]
                    for i_prime in self.var_nodes[j]:
                        if i_prime != i:
                            sum_r += R[i_prime, j]
                    Q_new[i, j] = sum_r
            
            Q = Q_new
            
            # Early termination check
            if early_term:
                # Compute total LLR for each bit
                L_total = Lc.copy()
                for j in range(self.n):
                    for i in self.var_nodes[j]:
                        L_total[j] += R[i, j]
                
                # Hard decision
                x_hat = (L_total < 0).astype(int)
                
                # Check parity
                if np.all((self.H @ x_hat) % 2 == 0):
                    return x_hat, n_iter
        
        # Final decision if not terminated early
        L_total = Lc.copy()
        for j in range(self.n):
            for i in self.var_nodes[j]:
                L_total[j] += R[i, j]
        
        x_hat = (L_total < 0).astype(int)
        return x_hat, self.max_iter
def decoder_classic(H,sigma,y,max_iter2,x):
    # llr = compute_llr(y, snr_db=snr_db2)

    # # Decode
    # decoded = belief_propagation(H, llr, max_iter=max_iter2)
    # num_bit_errors = np.sum( x!= decoded)
    decoder = SumProductDecoder(H,max_iter2)
    x_hat, n_iter = decoder.decode(y, sigma)
    #print(x_hat.shape)
    num_bit_errors=np.sum(x != x_hat)
    return x_hat,num_bit_errors

##################################
def decoder_classic_polar(H,sigma,y,max_iter2,x):
    snr_db_1=10*math.log10(sigma)
    llr = compute_llr(y, snr_db=snr_db_1)

    
    x_hat= belief_propagation(H, llr, max_iter=100)


    num_bit_errors = np.sum(x != xhat)
    return x_hat,num_bit_errors


############################################################

##################################################################################################################
##################################################################################################################
##################################################################################################################

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch ECCT')
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--workers', type=int, default=0)#4
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--gpus', type=str, default='-1', help='gpus ids')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--test_batch_size', type=int, default=1)#2048
    parser.add_argument('--test_records', type=int, default=4000)
    parser.add_argument('--seed', type=int, default=42)

    # Code args
    parser.add_argument('--code_type', type=str, default='LDPC',
                        choices=['BCH', 'POLAR', 'LDPC', 'CCSDS', 'MACKAY'])
    parser.add_argument('--code_k', type=int, default=60)
    parser.add_argument('--code_n', type=int, default=121)
    parser.add_argument('--standardize', action='store_true')

    # model args
    parser.add_argument('--N_dec', type=int, default=6)
    parser.add_argument('--d_model', type=int, default=32)
    parser.add_argument('--h', type=int, default=8)

    args = parser.parse_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    set_seed(args.seed)
    
    ####################################################################

    class Code():
        pass
    code = Code()
    code.k = args.code_k
    code.n = args.code_n
    code.code_type = args.code_type
    G, H = Get_Generator_and_Parity(code,standard_form=args.standardize)
    code.generator_matrix = torch.from_numpy(G).transpose(0, 1).long()
    code.pc_matrix = torch.from_numpy(H).long()
    args.code = code
    ####################################################################
    model_dir = os.path.join('Results_ECCT',
                             args.code_type + '__Code_n_' + str(
                                 args.code_n) + '_k_' + str(
                                 args.code_k) + '__' + datetime.now().strftime(
                                 "%d_%m_%Y_%H_%M_%S"))
    os.makedirs(model_dir, exist_ok=True)
    args.path = model_dir
    handlers = [
        logging.FileHandler(os.path.join(model_dir, 'logging.txt'))]
    handlers += [logging.StreamHandler()]
    logging.basicConfig(level=logging.INFO, format='%(message)s',
                        handlers=handlers)
    logging.info(f"Path to model/logs: {model_dir}")
    logging.info(args)
    
    args.EbNo_range_test=range(4,7)#(4,7)
    
    EbNo_range_test =args.EbNo_range_test

    output_test_nueral,test_dataset_nueral=main(args)
    
    Rate=code.k/code.n
    sigma2=[math.sqrt((1/(2*Rate))*(10**(-1*ii/10))) for ii in EbNo_range_test ]
    # std_test = [2*(code.k/code.n)*ii for ii in EbNo_range_test]
    # snr_db_list=[10*math.log10(ii) for ii in std_test ]
    # G1 = torch.load('G_POLAR64,32.pt').numpy().astype(int)  # Shape (64,32)
    # H1 = torch.load('H_POLAR64,32.pt').numpy().astype(int)  # Shape (32,64)
    # G=np.transpose(G1)
    # H=np.transpose(H1)
    
    total_test_time_1=time.time()
   
    out1=[]
    out2=[]
    out3=[]
    ##############
    Haming_distance_neural_sigma1=[]
    Haming_distance_neural_sigma2=[]
    Haming_distance_neural_sigma3=[]
    ################
    Haming_distance_classic_sigma1=[]
    Haming_distance_classic_sigma2=[]
    Haming_distance_classic_sigma3=[]
    
    #######################################
    mmd_g_nueral_sigma1=[]
    mmd_g_nueral_sigma2=[]
    mmd_g_nueral_sigma3=[]
    #############################
    mmd_g_classic_sigma1=[]
    mmd_g_classic_sigma2=[]
    mmd_g_classic_sigma3=[]
    
    ##########################################
    mmd_p_nueral_sigma1=[]
    mmd_p_nueral_sigma2=[]
    mmd_p_nueral_sigma3=[]
    ##################################
    mmd_p_classic_sigma1=[]
    mmd_p_classic_sigma2=[]
    mmd_p_classic_sigma3=[]
    ####################################################
    Ferchet_dataset_matrix_nural_X_sigma1=np.zeros((args.test_records,code.n))
    Ferchet_dataset_matrix_nural_X_sigma2=np.zeros((args.test_records,code.n))
    Ferchet_dataset_matrix_nural_X_sigma3=np.zeros((args.test_records,code.n))
    
    Ferchet_dataset_matrix_nural_Y_sigma1=np.zeros((args.test_records,code.n))
    Ferchet_dataset_matrix_nural_Y_sigma2=np.zeros((args.test_records,code.n))
    Ferchet_dataset_matrix_nural_Y_sigma3=np.zeros((args.test_records,code.n))
    
    # Ferchet_dataset_matrix_classic_X_sigma1=np.zeros((args.test_records,code.n))
    # Ferchet_dataset_matrix_classic_X_sigma1=np.zeros((args.test_records,code.n))
    # Ferchet_dataset_matrix_classic_X_sigma1=np.zeros((args.test_records,code.n))
    
    
    Ferchet_dataset_matrix_classic_Y_sigma1=np.zeros((args.test_records,code.n))
    Ferchet_dataset_matrix_classic_Y_sigma2=np.zeros((args.test_records,code.n))
    Ferchet_dataset_matrix_classic_Y_sigma3=np.zeros((args.test_records,code.n))
    
    print(args.code.n)
    
    
    ##########################################
    max_iter=50
    
    for ii in range(3):
        tt=output_test_nueral[ii]
        for j in range(args.test_records):
             bb=tt[j]
             bb2=bb['recived'].numpy().T
             bb3=bb['x_bin'].numpy().T
             bb4=bb['Decode'].numpy().T
             bb5=bb['Haming_distance']
            # print(f'x_bin.shape={bb3.shape}, Decode_nural.shape={bb4.shape}')
             nural_mmd_g= compute_mmd(bb3, bb4, kernel='rbf', sigma=1.0)
            # print(f'nural_mmd_g={nural_mmd_g}')
             nural_mmd_p = compute_mmd(bb3, bb4, kernel='poly', degree=3, coef0=1)
             
           
             # time_classic_1=time.time()
             classic_decode,Hamming_classic=decoder_classic(H,sigma2[ii],bb2,max_iter,bb3)
             # time_classic_2=time.time()
             # time_classic_duration=time_classic_2-time_classic_1
             # print(f'time_classic_duration={time_classic_duration}')
             #print(f'classic_decode.shpae={classic_decode.shape}')
             classic_mmd_g= compute_mmd(bb3, classic_decode, kernel='rbf', sigma=1.0)
             classic_mmd_p = compute_mmd(bb3, classic_decode, kernel='poly', degree=3, coef0=1)
             
             #print(f'Hamming_classic:={Hamming_classic}, Haming_nueral:={bb5}')
             if ii==0:
                 Ferchet_dataset_matrix_nural_X_sigma1[j,:]=bb3.T
                 Ferchet_dataset_matrix_nural_Y_sigma1[j,:]=bb4.T 
                 
                 Ferchet_dataset_matrix_classic_Y_sigma1[j,:]=classic_decode.T
                 
                 out1.append({ 'x_bin_1': bb3,
                  'recived': bb2,
                  'peredict_nural':bb4,
                  'peredict_classic':classic_decode,
                  'Haming_nural':bb5,
                  'Hamming_classic':Hamming_classic
                  })
                 Haming_distance_neural_sigma1.append(bb5)
                 Haming_distance_classic_sigma1.append(Hamming_classic)
                 mmd_g_nueral_sigma1.append(nural_mmd_g)
                 mmd_g_classic_sigma1.append(classic_mmd_g)
                 mmd_p_nueral_sigma1.append(nural_mmd_p)
                 mmd_p_classic_sigma1.append(classic_mmd_p)
                 
                 
                 
             elif ii==1:
                 
                Ferchet_dataset_matrix_nural_X_sigma2[j,:]=bb3.T
                Ferchet_dataset_matrix_nural_Y_sigma2[j,:]=bb4.T
                
                Ferchet_dataset_matrix_classic_Y_sigma2[j,:]=classic_decode.T
                out2.append({ 'x_bin_1': bb3,
                 'recived': bb2,
                 'peredict_nural':bb4,
                 'peredict_classic':classic_decode,
                 'Haming_nural':bb5,
                 'Hamming_classic':Hamming_classic
                 })
                Haming_distance_neural_sigma2.append(bb5)
                Haming_distance_classic_sigma2.append(Hamming_classic)
                mmd_g_nueral_sigma2.append(nural_mmd_g)
                mmd_g_classic_sigma2.append(classic_mmd_g)
                mmd_p_nueral_sigma2.append(nural_mmd_p)
                mmd_p_classic_sigma2.append(classic_mmd_p)
             else:
                 
                 Ferchet_dataset_matrix_nural_X_sigma3[j,:]=bb3.T
                 Ferchet_dataset_matrix_nural_Y_sigma3[j,:]=bb4.T
                 
                 Ferchet_dataset_matrix_classic_Y_sigma3[j,:]=classic_decode.T
                 out3.append({ 'x_bin_1': bb3,
                  'recived': bb2,
                  'peredict_nural':bb4,
                  'peredict_classic':classic_decode,
                  'Haming_nural':bb5,
                  'Hamming_classic':Hamming_classic
                  })
                 
                 
                 Haming_distance_neural_sigma3.append(bb5)
                 Haming_distance_classic_sigma3.append(Hamming_classic)
                 mmd_g_nueral_sigma3.append(nural_mmd_g)
                 mmd_g_classic_sigma3.append(classic_mmd_g)
                 mmd_p_nueral_sigma3.append(nural_mmd_p)
                 mmd_p_classic_sigma3.append(classic_mmd_p)
    
                 
    print('################################################')
    print('#############Reasults##########################')
    print('################################################')
    print('####BER_neural###########################')
    f = open(f'output_{code.code_type}_n_{code.n}_k_{code.k}_testsize_{args.test_records}.txt', 'a')
    
    BER_mean_nural_sigma1=sum(Haming_distance_neural_sigma1)/(code.n*args.test_records)
    print(f'BER_mean_nural_sigma1={BER_mean_nural_sigma1}')
    f.write(f'BER_mean_nural_sigma1={BER_mean_nural_sigma1}\n')
    BER_mean_nural_sigma2=sum(Haming_distance_neural_sigma2)/(code.n*args.test_records)
    print(f'BER_mean_nural_sigma2={BER_mean_nural_sigma2}')
    f.write(f'BER_mean_nural_sigma2={BER_mean_nural_sigma2}\n')
    BER_mean_nural_sigma3=sum(Haming_distance_neural_sigma3)/(code.n*args.test_records)
    print(f'BER_mean_nural_sigma3={BER_mean_nural_sigma3}')
    f.write(f'BER_mean_nural_sigma3={BER_mean_nural_sigma3}\n')
    
    print('####BER_Sum-Product###########################')
    BER_mean_classic_sigma1=sum(Haming_distance_classic_sigma1)/(code.n*args.test_records)
    print(f'BER_mean_classic_sigma1={BER_mean_classic_sigma1}')
    f.write(f'BER_mean_classic_sigma1={BER_mean_classic_sigma1}\n')
    BER_mean_classic_sigma2=sum(Haming_distance_classic_sigma2)/(code.n*args.test_records)
    print(f'BER_mean_classic_sigma2={BER_mean_classic_sigma2}')
    f.write(f'BER_mean_classic_sigma2={BER_mean_classic_sigma2}\n')
    BER_mean_classic_sigma3=sum(Haming_distance_classic_sigma3)/(code.n*args.test_records)
    print(f'BER_mean_classic_sigma3={BER_mean_classic_sigma3}')
    f.write(f'BER_mean_classic_sigma3={BER_mean_classic_sigma3}\n')
    print('####MMD-Guassian-neural###########################')
    mmd_mean_g_nueral_sigma1=sum(mmd_g_nueral_sigma1)/(args.test_records)
    print(f'mmd_mean_g_neural_sigma1={mmd_mean_g_nueral_sigma1}')
    f.write(f'mmd_mean_g_neural_sigma1={mmd_mean_g_nueral_sigma1}\n')
    mmd_mean_g_nueral_sigma2=sum(mmd_g_nueral_sigma2)/(args.test_records)
    print(f'mmd_mean_g_neural_sigma2={mmd_mean_g_nueral_sigma2}')
    f.write(f'mmd_mean_g_neural_sigma2={mmd_mean_g_nueral_sigma2}\n')
    mmd_mean_g_nueral_sigma3=sum(mmd_g_nueral_sigma3)/(args.test_records)
    print(f'mmd_mean_g_neural_sigma3={mmd_mean_g_nueral_sigma3}')
    f.write(f'mmd_mean_g_neural_sigma3={mmd_mean_g_nueral_sigma3}\n')
    print('####MMD-Guassian-Sum-Product###########################')
    mmd_mean_g_classic_sigma1=sum(mmd_g_classic_sigma1)/(args.test_records)
    print(f'mmd_mean_g_classic_sigma1={mmd_mean_g_classic_sigma1}')
    f.write(f'mmd_mean_g_classic_sigma1={mmd_mean_g_classic_sigma1}\n')
    mmd_mean_g_classic_sigma2=sum(mmd_g_classic_sigma2)/(args.test_records)
    print(f'mmd_mean_g_classic_sigma2={mmd_mean_g_classic_sigma2}')
    f.write(f'mmd_mean_g_classic_sigma2={mmd_mean_g_classic_sigma2}\n')
    mmd_mean_g_classic_sigma3=sum(mmd_g_classic_sigma3)/(args.test_records)
    print(f'mmd_mean_g_classic_sigma3={mmd_mean_g_classic_sigma3}')
    f.write(f'mmd_mean_g_classic_sigma3={mmd_mean_g_classic_sigma3}\n')
    print('####MMD-Polynomial-neural###########################')
    mmd_mean_p_nueral_sigma1=sum(mmd_p_nueral_sigma1)/(args.test_records)
    print(f'mmd_mean_p_neural_sigma1={mmd_mean_p_nueral_sigma1}')
    f.write(f'mmd_mean_p_neural_sigma1={mmd_mean_p_nueral_sigma1}\n')
    mmd_mean_p_nueral_sigma2=sum(mmd_p_nueral_sigma2)/(args.test_records)
    print(f'mmd_mean_p_neural_sigma2={mmd_mean_p_nueral_sigma2}')
    f.write(f'mmd_mean_p_neural_sigma2={mmd_mean_p_nueral_sigma2}\n')
    mmd_mean_p_nueral_sigma3=sum(mmd_p_nueral_sigma3)/(args.test_records)
    print(f'mmd_mean_p_neural_sigma3={mmd_mean_p_nueral_sigma3}')
    f.write(f'mmd_mean_p_neural_sigma3={mmd_mean_p_nueral_sigma3}\n')
    print('####MMD-Polynomial-Sum-Product###########################')
    mmd_mean_p_classic_sigma1=sum(mmd_p_classic_sigma1)/(args.test_records) 
    print(f'mmd_mean_p_classic_sigma1={mmd_mean_p_classic_sigma1}')
    f.write(f'mmd_mean_p_classic_sigma1={mmd_mean_p_classic_sigma1}\n') 
    mmd_mean_p_classic_sigma2=sum(mmd_p_classic_sigma2)/(args.test_records)
    print(f'mmd_mean_p_classic_sigma2={mmd_mean_p_classic_sigma2}')
    f.write(f'mmd_mean_p_classic_sigma2={mmd_mean_p_classic_sigma2}\n')
    mmd_mean_p_classic_sigma3=sum(mmd_p_classic_sigma3)/(args.test_records)
    print(f'mmd_mean_p_classic_sigma3={mmd_mean_p_classic_sigma3}')
    f.write(f'mmd_mean_p_classic_sigma3={mmd_mean_p_classic_sigma3}\n')
    ##################################################
    print('####Ferchet Distance-neuaral###########################')
    frechet_nural_sigma1=frechet_distance_numpy(Ferchet_dataset_matrix_nural_X_sigma1,Ferchet_dataset_matrix_nural_Y_sigma1 , eps=1e-10)
    print(f'frechet_nural_sigma1={frechet_nural_sigma1}')
    f.write(f'frechet_nural_sigma1={frechet_nural_sigma1}\n')
    frechet_nural_sigma2=frechet_distance_numpy(Ferchet_dataset_matrix_nural_X_sigma2,Ferchet_dataset_matrix_nural_Y_sigma2 , eps=1e-10)
    print(f'frechet_nural_sigma2={frechet_nural_sigma2}')
    f.write(f'frechet_nural_sigma2={frechet_nural_sigma2}\n')
    frechet_nural_sigma3=frechet_distance_numpy(Ferchet_dataset_matrix_nural_X_sigma3,Ferchet_dataset_matrix_nural_Y_sigma3 , eps=1e-10)
    print(f'frechet_nural_sigma3={frechet_nural_sigma3}')
    f.write(f'frechet_nural_sigma3={frechet_nural_sigma3}\n')
    print('####Ferchet Distance-Sum-Product###########################')
    frechet_classic_sigma1=frechet_distance_numpy(Ferchet_dataset_matrix_nural_X_sigma1,Ferchet_dataset_matrix_classic_Y_sigma1 , eps=1e-10)
    print(f'frechet_classic_sigma1={frechet_classic_sigma1}')
    f.write(f'frechet_classic_sigma1={frechet_classic_sigma1}\n')
    frechet_classic_sigma2=frechet_distance_numpy(Ferchet_dataset_matrix_nural_X_sigma2,Ferchet_dataset_matrix_classic_Y_sigma2 , eps=1e-10)
    print(f'frechet_classic_sigma2={frechet_classic_sigma2}')
    f.write(f'frechet_classic_sigma2={frechet_classic_sigma2}\n')
    frechet_classic_sigma3=frechet_distance_numpy(Ferchet_dataset_matrix_nural_X_sigma3,Ferchet_dataset_matrix_classic_Y_sigma3 , eps=1e-10)
    print(f'frechet_classic_sigma3={frechet_classic_sigma3}')
    f.write(f'frechet_classic_sigma3={frechet_classic_sigma3}\n')

    ################################################
    
    threshold = 1e-6
    # print('gook')
    
    MM11=Ferchet_dataset_matrix_nural_Y_sigma1
    # print('toos')
    MM12=Ferchet_dataset_matrix_classic_Y_sigma1
    # print('gookk')
    
    MM_sigma1=MM11.T @ MM11-MM12.T@ MM12
    # print('gook2')
    eigenvalues_sigma1 = np.linalg.eigvalsh(MM_sigma1)
    # print('gook3')
    norm_sigma1=np.linalg.norm(eigenvalues_sigma1)
    # print('gook4')   
    small_elements_sigma1 = np.abs(eigenvalues_sigma1) < threshold
    # print('gook5')


    percent_sigma1 = 100 * np.sum(small_elements_sigma1) / eigenvalues_sigma1.size
    # print('gook6')

    print('#######results for second part##########')
    print(f'L2 Norm of eigenvalues_sigma1={norm_sigma1}' )
    f.write(f'L2 Norm of eigenvalues_sigma1={norm_sigma1}\n')
    print(f'Percentage of elements with |value| < {threshold} for sigma1= {percent_sigma1}')
    f.write(f'Percentage of elements with |value| < {threshold} for sigma1= {percent_sigma1}\n')

    MM21=Ferchet_dataset_matrix_nural_Y_sigma2
    MM22=Ferchet_dataset_matrix_classic_Y_sigma2
     
    MM_sigma2=MM21.T @ MM21-MM22.T@ MM22
    eigenvalues_sigma2 = np.linalg.eigvalsh(MM_sigma2)
    
    norm_sigma2=np.linalg.norm(eigenvalues_sigma2)
   
    small_elements_sigma2 = np.abs(eigenvalues_sigma2) < threshold


    percent_sigma2 = 100 * np.sum(small_elements_sigma2) / eigenvalues_sigma2.size


    print(f'L2 Norm of eigenvalues_sigma2={norm_sigma2}' )
    f.write(f'L2 Norm of eigenvalues_sigma2={norm_sigma2}\n' )
    print(f'Percentage of elements with |value| < {threshold} for sigma2= {percent_sigma2}')
    f.write(f'Percentage of elements with |value| < {threshold} for sigma2= {percent_sigma2}\n')
    MM31=Ferchet_dataset_matrix_nural_Y_sigma3
    MM32=Ferchet_dataset_matrix_classic_Y_sigma3
     
    MM_sigma3=MM31.T @ MM31-MM32.T@ MM32
    eigenvalues_sigma3 = np.linalg.eigvalsh(MM_sigma3)
    
    norm_sigma3=np.linalg.norm(eigenvalues_sigma3)
   
    small_elements_sigma3 = np.abs(eigenvalues_sigma3) < threshold


    percent_sigma3 = 100 * np.sum(small_elements_sigma3) / eigenvalues_sigma3.size


    print(f'L2 Norm of eigenvalues_sigma3={norm_sigma3}' )
    f.write(f'L2 Norm of eigenvalues_sigma3={norm_sigma3}\n' )
    print(f'Percentage of elements with |value| < {threshold} for sigma3= {percent_sigma3}')
    f.write(f'Percentage of elements with |value| < {threshold} for sigma3= {percent_sigma3}\n')
    v1_abs_1 = np.abs(eigenvalues_sigma1)
    v1_abs=v1_abs_1[(v1_abs_1 >= 0) & (v1_abs_1 <= 200)]
    v2_abs_2 = np.abs(eigenvalues_sigma2)
    v2_abs=v2_abs_2[(v2_abs_2 >= 0) & (v2_abs_2 <= 200)]
    v3_abs_3 = np.abs(eigenvalues_sigma3)
    v3_abs=v3_abs_3[(v3_abs_3 >= 0) & (v3_abs_3 <= 200)]
    plt.figure(1)
    plt.hist(v1_abs, bins=50, color='blue', alpha=0.7)
    # plt.title('Histogram of |eigenvalues of M| for E_b/N0=4')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.savefig(f'Hist_{code.code_type}_n_{code.n}_k_{code.k}_testsize_{args.test_records}_EbN0=4.png', bbox_inches='tight')

    np.save(f'eig_{code.code_type}_n_{code.n}_k_{code.k}_testsize_{args.test_records}_EbN0=4.npy', v1_abs)
    
    plt.figure(2)
    plt.hist(v2_abs, bins=50, color='blue', alpha=0.7)
    # plt.title('Histogram of |eigenvalues of M | for E_b/N0=5')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.savefig(f'Hist_{code.code_type}_n_{code.n}_k_{code.k}_testsize_{args.test_records}_EbN0=5.png', bbox_inches='tight')
    np.save(f'eig_{code.code_type}_n_{code.n}_k_{code.k}_testsize_{args.test_records}_EbN0=5.npy', v2_abs)
    plt.figure(3)
    plt.hist(v3_abs, bins=50, color='blue', alpha=0.7)
    # plt.title('Histogram of |eigenvalues of M| for E_b/N0=6')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.savefig(f'Hist_{code.code_type}_n_{code.n}_k_{code.k}_testsize_{args.test_records}_EbN0=6.png', bbox_inches='tight')
    np.save(f'eig_{code.code_type}_n_{code.n}_k_{code.k}_testsize_{args.test_records}_EbN0=6.npy', v3_abs)
    total_test_time_2=time.time()
    
    total_test_time_duration=total_test_time_2-total_test_time_1
    print(f'total_test_time_duration_minutes={total_test_time_duration/60}')
    f.write(f'total_test_time_duration_minutes={total_test_time_duration/60}\n')
    print(f'total_test_time_duration_minutes={total_test_time_duration/(60*args.test_records)}(persample:samples={args.test_records})')
    f.write(f'total_test_time_duration_minutes={total_test_time_duration/(60*args.test_records)}(persample:samples={args.test_records})\n')
    f.close()


  
    
    
   
        
        