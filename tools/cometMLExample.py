


from comet_ml import Experiment

#create an experiment

if __name__ == '__main__':
    experiment = Experiment(api_key="v063r9jHG5GDdPFvCtsJmHYZu",
                        project_name="general", workspace="glenb")
    import keras
    import numpy as np
    batch_size = 128
    
    experiment.log_parameter("batch_size", 128)
    
    for i in range(51):
        metrics = {
        "data": np.random.normal(0,1,1)[0],
        "list": np.random.uniform(0,1,1)[0]
         }
        experiment.log_metrics(metrics)
        experiment.log_epoch_end(i, step=i)
        
        
    experiment.end()