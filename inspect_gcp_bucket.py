from google.cloud import storage


if __name__ == "__main__":

    storage_client = storage.Client()
    bucket = storage_client.get_bucket("rl-framework-cluster-bucket")

    blobs = list(bucket.list_blobs(prefix=""))
    blobs = {x.name: x for x in blobs}

    candidates = [x for x in blobs if "/doodad/logs/" in x]
    a = blobs[candidates[-1]]
    print(a.download_as_string().decode("utf-8"))
