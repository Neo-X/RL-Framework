


from comet_ml import Experiment

#create an experiment

if __name__ == '__main__':
    experiment = Experiment(api_key="v063r9jHG5GDdPFvCtsJmHYZu",
                        project_name="general", workspace="glenb")
    experiment.add_tag("comet_test")
    experiment.set_name("comet_test")
    # experiment.log_dependency(self, "terrainRLAdapter", version)
    experiment.set_filename(fname="cometML_test")
    
    import keras
    import numpy as np
    batch_size = 128
    
    experiment.log_parameter("batch_size", 128)
    
    for i in range(6):
        
        experiment.set_step(step=i)
        for j in range(10):
            metrics = {
            "data": np.random.normal(i,1,1)[0],
            "list": np.random.uniform(j,1,1)[0]
             }
            experiment.log_metrics(metrics)
        ### Needs to be used so data can be ploted over x-axis
        # experiment.log_epoch_end(i, step=i)
        
        
    experiment.end()