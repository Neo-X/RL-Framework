from google.cloud import storage
import argparse


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", type=str, default="/doodad/logs2/")
    parser.add_argument("--bucket", type=str, default="rl-framework-cluster-bucket")
    args = parser.parse_args()

    storage_client = storage.Client()
    bucket = storage_client.get_bucket(args.bucket)

    blobs = list(bucket.list_blobs(prefix=""))
    blobs = {x.name: x for x in blobs}

    candidates = [x for x in blobs if args.logdir in x]
    for c in candidates:
        print(blobs[c].download_as_string().decode("utf-8"))
        input("press enter to continue...")
