import torch
from src.metrics import metrics_display,metrics_dict_creator

@torch.inference_mode()
def evaluate(encoder,head,loader,device):


    y_result_accumulated=torch.tensor([]).to(device)
    y_real_accumulated=torch.tensor([]).to(device)
    for x,y in loader:
        x,y=x.to(device),y.to(device)
        with torch.no_grad():
            h,_=encoder(x)
            y_result=head(h).squeeze(1)
            y_result=torch.sigmoid(y_result)
        y_result_accumulated=torch.cat([y_result_accumulated,y_result])
        y_real_accumulated=torch.cat([y_real_accumulated,y]).long()
        print("|",end=" ")
    print()
    metrics_display(y_real=y_real_accumulated,y_result=y_result_accumulated)
    metrics_dict=metrics_dict_creator(y_real=y_real_accumulated,y_result=y_result_accumulated)
    # reliability_plot(y_real=y_real_accumulated,y_result=y_result_accumulated,save_img=True,title=f"Reliability-curve_p{lp}_eval_data")
    return y_result_accumulated,y_real_accumulated,metrics_dict
