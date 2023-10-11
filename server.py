import flwr as fl
from flwr.common import Metrics
from typing import Callable, Dict, Optional, Tuple, List

from client import (
    load_data,
    PeftClient,
    client_fn,
    test,
)
    
from typing import List, Tuple
from torch.utils.data import DataLoader
import torch
from peft import (
    get_peft_config,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
    PeftType,
    PrefixTuningConfig,
    PromptEncoderConfig,
    PromptTuningConfig,
    PeftModel
)
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup, set_seed
from typing import Callable, Dict, Optional, Tuple

GLOBAL_MODEL_NAME_OR_PATH = "bert-base-uncased"
GLOBAL_DEVICE = torch.device("cuda")
    
trainloader, testloader = load_data(0)
globalConfig = PromptTuningConfig(task_type="SEQ_CLS", num_virtual_tokens=20)
globalModel = AutoModelForSequenceClassification.from_pretrained(GLOBAL_MODEL_NAME_OR_PATH, return_dict=True)
globalModel = get_peft_model(globalModel, globalConfig).to(GLOBAL_DEVICE)
globalModel = PeftClient(globalModel, trainloader, testloader, 0)

def get_evaluate_fn() -> Callable[[fl.common.NDArrays], Optional[Tuple[float, float]]]:
    """Return an evaluation function for centralized evaluation."""

    def evaluate(server_round:int, weights: fl.common.NDArrays, config) -> Optional[Tuple[float, float]]:
        """Use the entire CIFAR-10 test set for evaluation."""
        globalModel.set_parameters(weights)
        
        loss, accuracy = test(globalModel.model, testloader)
        
        return loss, {"accuracy": accuracy}
        

    return evaluate

# Define metric aggregation function
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    #print(metrics)
    
    num_total_evaluation_examples = sum([num_examples for num_examples, _ in metrics])
    print(metrics)
    
    print(f"Weighted Average Accuracy is {sum(accuracies) / sum(examples)}")
    #print(sum(weighted_losses) / num_total_evaluation_examples)

    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples)}

if __name__ == "__main__":
    # Define strategy
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        evaluate_metrics_aggregation_fn=weighted_average,
        evaluate_fn = get_evaluate_fn()
    )

    # Start server
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=20),
        strategy=strategy,
    )
