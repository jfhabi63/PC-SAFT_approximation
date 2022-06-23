from scripts_execute import *

def main():
    torch.manual_seed(2)
    X_input,comp = load_raw_data()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = Dataset_input(X_input,comp)

    loader_kwargs = {
        "batch_size": 256,
        "drop_last": False,
        "num_workers": 0,
        "pin_memory": False,
        "persistent_workers": False,
        "shuffle": False
    }

    data_loader = DataLoader(dataset=dataset,
                             **loader_kwargs
                             )
    name = "PC-SAFT_approx"
    model_act = torch.load('./active_model/' + name).to(device)
    model_act.eval()
    y_predicted = calc_pred(data_loader,model_act)
    X_input = retransform_X(X_input)
    output = pd.DataFrame(data=X_input, columns=["m_seg","sig","eps","m_seg_ML","sig_ML","eps_ML"])
    output["AARD_rho_pred"] = y_predicted.iloc[:,0]
    output["RMSE_rho_pred"] = y_predicted.iloc[:,1]
    output.to_csv("./active_model/output.csv",sep=";")
main()

