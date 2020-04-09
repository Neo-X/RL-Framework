import ray


if __name__ == "__main__":
    
    # ray.init(redis_address="deepspace9.Banatao.Berkeley.EDU:6379")
    ray.init(redis_address="169.229.222.227:6379")
    print (ray.get_webui_url())
    print (ray.cluster_resources())
    ray.shutdown()
    print("Done")