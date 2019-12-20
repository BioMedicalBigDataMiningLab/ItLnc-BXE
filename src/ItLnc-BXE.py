
import os
import csv
import time
import random
import argparse
import numpy as np
import pandas as pd
from Bio import SeqIO, SeqUtils
from xgboost.sklearn import XGBClassifier
from sklearn.externals import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score,accuracy_score,recall_score,f1_score,precision_score,precision_recall_curve,roc_curve,auc,matthews_corrcoef
from sklearn.model_selection import cross_val_score,train_test_split,StratifiedKFold
from deap import creator, base, tools, algorithms
import feamodule.ProtParam as PP
import feamodule.ORF_length as leng
from feamodule.CTD import CTD

##################################################
#                Model Reference                 #
#         a : Arabidopsis_thaliana               #
#         c : Chlamydomonas_reinhardtii          #
#         h : Hordeum_vulgare                    #
#         o : Oryza_sativa_Japonica_Group        #
#         p : Physcomitrella_patens              #
#         s : Solanum_tuberosum                  #
##################################################

model_reference={'a':['./Arabidopsis_thaliana_model/',
                      './Arabidopsis_thaliana_model/At_hexamer_table.tsv',
                      './Arabidopsis_thaliana_model/At_selected_features.npy',
                     './Arabidopsis_thaliana_model/At_model_params.txt'],
                 'c':['./Chlamydomonas_reinhardtii_model/',
                      './Chlamydomonas_reinhardtii_model/Cr_hexamer_table.tsv',
                      './Chlamydomonas_reinhardtii_model/Cr_selected_features.npy',
                     './Chlamydomonas_reinhardtii_model/Cr_model_params.txt'],
                'h':['./Hordeum_vulgare_model/',
                      './Hordeum_vulgare_model/Hv_hexamer_table.tsv',
                      './Hordeum_vulgare_model/Hv_selected_features.npy',
                     './Hordeum_vulgare_model/Hv_model_params.txt'],
                'o':['./Oryza_sativa_Japonica_Group_model/',
                      './Oryza_sativa_Japonica_Group_model/Os_hexamer_table.tsv',
                      './Oryza_sativa_Japonica_Group_model/Os_selected_features.npy',
                     './Oryza_sativa_Japonica_Group_model/Os_model_params.txt'],
                'p':['./Physcomitrella_patens_model/',
                      './Physcomitrella_patens_model/Pp_hexamer_table.tsv',
                      './Physcomitrella_patens_model/Pp_selected_features.npy',
                     './Physcomitrella_patens_model/Pp_model_params.txt'],
                's':['./Solanum_tuberosum_model/',
                      './Solanum_tuberosum_model/St_hexamer_table.tsv',
                      './Solanum_tuberosum_model/St_selected_features.npy',
                     './Solanum_tuberosum_model/St_model_params.txt']}


def main():
    parser = argparse.ArgumentParser(description='ItLnc-BXE: Prediction 0f plant lncRNA by using a Bagging-XGBoost-ensemble method with multiple features')

    parser.add_argument('-i','--input',dest="input",type=str,help="input sequence in fasta format(required)")
    parser.add_argument('-m','--model',dest="model",type=str,help="use [a](Arabidopsis_thaliana) or [c](Chlamydomonas_reinhardtii) or [h](Hordeum_vulgare) or [o](Oryza_sativa_Japonica_Group) or [p](Physcomitrella_patens) or [s](Solanum_tuberosum) model")
    parser.add_argument('-o','--output',dest="output",type=str,help="output file format like [Sequence Class]")
    args = parser.parse_args()

    if args.input and args.output and args.model:
        input_file = args.input
        output_file = args.output
        model = args.model
        print("[INFO] Getting all features from transcripts...")
        features,seq_id = get_all_features(input_file,model)
        print("[INFO] Done!")
        print("[INFO] Data processing...")
        data=data_processing(features,model)
        print("[INFO] Done!")
        print("[INFO] Predicting...")
        label = predict(data,model,5)
        print("[INFO] Done!")
        print("[INFO] Saving labels...")
        save_result(label,output_file,seq_id)
        print("Run successful!")
        print("There were " + str(np.sum(label==1)) + " lncRNAs predicted from " + str(len(label))+" transcripts!")
    else:
        print("Please check your --input and --output and --model parameters!")
    return 0


def get_Seq_ORF_features(file_path,input_file,model):
    seq_id = []  
    features_dict = {}
    transcript_sequences = []
    for record in SeqIO.parse(input_file, "fasta"):
        name = record.id
        name = name.lower()
        seq_id.append(name)
        seq = record.seq
        transcript_sequences.append(seq)
        features_dict[name] = {}
        features_dict[name]["length"] = len(record.seq)   
        G_C = SeqUtils.GC(record.seq)
        features_dict[name]["G+C"] = G_C
        insta_fe,PI_fe,gra_fe = PP.param(seq)
        Len,Cov,inte_fe = leng.len_cov(seq)
        features_dict[name].update({"ORF-integrity":inte_fe,"ORF-coverage":Cov,"Instability":insta_fe,"PI":PI_fe,"Gravy":gra_fe})
        A,T,G,C,AT,AG,AC,TG,TC,GC,A0,A1,A2,A3,A4,T0,T1,T2,T3,T4,G0,G1,G2,G3,G4,C0,C1,C2,C3,C4 = CTD(seq)
        features_dict[name].update({'A':A,'T':T,'G':G,'C':C,'AT':AT,'AG':AG,'AC':AC,'TG':TG,'TC':TC,'GC':GC,'A0':A0,'A1':A1,'A2':A2,'A3':A3,'A4':A4,'T0':T0,'T1':T1,'T2':T2,'T3':T3,'T4':T4,'G0':G0,'G1':G1,'G2':G2,'G3':G3,'G4':G4,'C0':C0,'C1':C1,'C2':C2,'C3':C3,'C4':C4})
    os.system("python3 "+file_path+"/feamodule/cpat.py -g "+input_file+" -o temp_cpat.txt -x "+model_reference[model][1])  #Use cpat to get fickett , hexamer , ORF
    with open("temp_cpat.txt.dat", "r") as tabular:
        cpat_reader = csv.reader(tabular, delimiter=("\t"))
        for row in cpat_reader:
            name = row[0]
            name = name.lower() 
            ORF = float(row[2]) 
            fickett = float(row[3])
            hexamer = float(row[4])
            features_dict[name]["ORF"] = ORF  
            features_dict[name]["fickett"] = fickett 
            features_dict[name]["hexamer"] = hexamer  
    os.system("rm temp_cpat.txt.dat")
    return features_dict,seq_id,transcript_sequences

def get_Coaon_bias_features(file_path,seq_id,transcript_sequences):
    temp1 = open("transcript_sequences.fa",'w')
    for line in transcript_sequences:
        temp1.write(str(line))
        temp1.write("\n")
    temp1.close()
    os.system("node "+file_path+"/feamodule/extractCodonBiasMeasures.js --input transcript_sequences.fa --output1 CodonBiasMeasures.csv")  #Download nodejs
    os.system("rm transcript_sequences.fa")
    conda_csv = pd.read_csv("CodonBiasMeasures.csv")
    conda_csv = conda_csv.T
    conda_csv= conda_csv.iloc[1:-1]
    del conda_csv[1]
    conda_columns = ['Fop', 'CUB', 'RCB', 'EW', 'SCUO', 'RSCU_TTT',
       'RSCU_TTC', 'RSCU_TTA', 'RSCU_TTG', 'RSCU_CTT', 'RSCU_CTC', 'RSCU_CTA',
       'RSCU_CTG', 'RSCU_ATT', 'RSCU_ATC', 'RSCU_ATA', 'RSCU_ATG', 'RSCU_GTT',
       'RSCU_GTC', 'RSCU_GTA', 'RSCU_GTG', 'RSCU_TCT', 'RSCU_TCC', 'RSCU_TCA',
       'RSCU_TCG', 'RSCU_CCT', 'RSCU_CCC', 'RSCU_CCA', 'RSCU_CCG', 'RSCU_ACT',
       'RSCU_ACC', 'RSCU_ACA', 'RSCU_ACG', 'RSCU_GCT', 'RSCU_GCC', 'RSCU_GCA',
       'RSCU_GCG', 'RSCU_TAT', 'RSCU_TAC', 'RSCU_CAT', 'RSCU_CAC', 'RSCU_CAA',
       'RSCU_CAG', 'RSCU_AAT', 'RSCU_AAC', 'RSCU_AAA', 'RSCU_AAG', 'RSCU_GAT',
       'RSCU_GAC', 'RSCU_GAA', 'RSCU_GAG', 'RSCU_TGT', 'RSCU_TGC', 'RSCU_TGG',
       'RSCU_CGT', 'RSCU_CGC', 'RSCU_CGA', 'RSCU_CGG', 'RSCU_AGT', 'RSCU_AGC',
       'RSCU_AGA', 'RSCU_AGG', 'RSCU_GGT', 'RSCU_GGC', 'RSCU_GGA', 'RSCU_GGG']
    conda_csv = pd.DataFrame(np.array(conda_csv),columns=conda_columns)
    conda_temp = dict(conda_csv.T)
    conda_features = {}
    for i in range(len(conda_csv)):
        temp2 = dict(conda_temp[i][conda_columns])
        conda_features[seq_id[i]]=temp2
        del temp2
    os.system("rm CodonBiasMeasures.csv")
    return conda_features

def get_Alignment_features(file_path,input_file,seq_id):
    #Get Alignment-based features , ORF score , trimers
    os.system("python2 "+file_path+"/feamodule/plncpro/plncpro_getfeatures.py -i "+input_file+" -p pred_res -o sample_preds -d "+file_path+"/feamodule/blast/blastdb/sprotdb -t 4")
    align_csv = pd.read_csv("plncpro_features", delimiter=("\t"),low_memory=False)
    del align_csv["Label"]
    del align_csv["seqid"]
    del align_csv["length"]
    del align_csv["orf_coverage"]
    align_columns = list(align_csv.columns)
    align_temp = dict(align_csv.T)
    alignment_features = {}
    for i in range(len(align_csv)):
        temp = dict(align_temp[i][align_columns])
        alignment_features[seq_id[i]]=temp
        del temp
    os.system("rm plncpro_features")
    return alignment_features

def get_all_features(input_file,model):
    file_path = os.path.dirname(os.path.abspath(__file__))
    seq_id = []  
    features_dict = {}
    transcript_sequences = []
    features_dict,seq_id,transcript_sequences=get_Seq_ORF_features(file_path,input_file,model)
    conda_features = get_Coaon_bias_features(file_path,seq_id,transcript_sequences)
    alignment_features=get_Alignment_features(file_path,input_file,seq_id)
    for name in seq_id:
        features_dict[name].update(conda_features[name])
        features_dict[name].update(alignment_features[name])
    features= pd.DataFrame(features_dict).T.reset_index(drop=False)
    features.to_csv("all_features.csv",index=False)
    return features,seq_id

def data_processing(features,model):
    selected_features = np.load(model_reference[model][2]).tolist()
    data = pd.DataFrame()
    for feature_id in selected_features:
        data = pd.concat([data,features[feature_id]],axis=1)
    data=np.array(data)
    return data

def predict(data,model,n_clf):
    estimators_X =[]
    for i in range(0,n_clf):
        xgb = joblib.load(model_reference[model][0]+'xgb_'+str(i)+'.pkl')
        score = xgb.predict_proba(data)
        estimators_X.append(score[:,1].tolist())
        del xgb
    lr = joblib.load(model_reference[model][0]+'lr.pkl')
    estimators_X = np.array(estimators_X)
    estimators_X = estimators_X.T
    label = lr.predict(estimators_X)
    return label

def save_result(label,output_file,seq_id):
    with open(output_file,'w') as out:
        out.write("Sequence"+'\t'+"Class"+'\n')
        count = 0
        for i in label:
            if i == 1:
                out.write(seq_id[count]+'\t'+'lncRNA'+'\n')
            else:
                out.write(seq_id[count]+'\t'+'pct'+'\n')
            count += 1

if __name__ == '__main__':
    main()
