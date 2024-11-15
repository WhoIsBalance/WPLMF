import argparse

def parameter_parser():

    parser = argparse.ArgumentParser()

    # hyperparameter for FinalDataset
    parser.add_argument('--n_drug',type=int,default=1177,help='number of drugs')
    parser.add_argument('--n_protein',type=int,default=1749,help='number of proteins')
    parser.add_argument('--n_adr',type=int,default=4247,help='number of adrs')
    parser.add_argument('--n_entity',type=int,default=2926,help='number of entities')
    parser.add_argument('--n_relation',type=int,default=49,help='number of relations')
    parser.add_argument('--sider_scr',type=str,default='..\\dataset\\final_dataset1\\sider.txt')
    parser.add_argument('--sider_mat',type=str,default='..\\dataset\\final_dataset1\\drug_se_mat.txt')
    parser.add_argument('--save_path', type=str ,default= '..\\save2\\fd')
    parser.add_argument('--kg_scr',type=str,default='..\\dataset\\final_dataset1\\kg.txt')
    parser.add_argument('--root', type=str, default='..\\dataset\\final_dataset1')

    # hyperparameter for SIDER4
    # parser.add_argument('--n_drug',type=int,default=1080,help='number of drugs')
    # parser.add_argument('--n_protein',type=int,default=1749,help='number of proteins')
    # parser.add_argument('--n_adr',type=int,default=5579,help='number of adrs')
    # parser.add_argument('--n_entity',type=int,default=2926,help='number of entities')
    # parser.add_argument('--sider_mat',type=str,default='..\\dataset\\new_sider4\\drug_se_mat.txt')
    # parser.add_argument('--save_path', type=str ,default= '..\\save\\sider4')
    # parser.add_argument('--enzyme_mat', type=str, default='..\\dataset\\new_sider4\\enzyme.txt')
    # parser.add_argument('--pathway_mat', type=str, default='..\\dataset\\new_sider4\\pathway.txt')
    # parser.add_argument('--target_mat', type=str, default='..\\dataset\\new_sider4\\target.txt')
    # parser.add_argument('--transporter_mat', type=str, default='..\\dataset\\new_sider4\\transporter.txt')
    # parser.add_argument('--treatment_mat', type=str, default='..\\dataset\\new_sider4\\treatment.txt')
    # parser.add_argument('--n_indication',type=int,default=1546,help='number of relations')
    # parser.add_argument('--root', type=str, default='..\\dataset\\new_sider4')


    #训练参数
    parser.add_argument('--k_fold', type=int, default=5)
    parser.add_argument('--thres', type=float, default=0.8)
    parser.add_argument('--n_basemodel', type=int, default=5)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--lr', type=float, default=0.02)
    

    # node2vec参数
    parser.add_argument('--n_walks', type=int, default=100)
    parser.add_argument('--walk_length', type=int, default=5)
    parser.add_argument('--pq', type=list, default=[(0.5,2),(2,0.5),(1,1),(0.7,1.5),(1.5, 0.7)])

    # CPL 训练参数
    parser.add_argument('--iter', type=int ,default=100)
    parser.add_argument('--epoch', type=int, default=50)
    parser.add_argument('--alpha', type=float, default=0.9)
    parser.add_argument('--gamma', type=float, default=2)


    # options
    parser.add_argument('--train_kg', type=bool, default=False)
    return parser.parse_args()