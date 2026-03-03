from minio import Minio
from minio.error import S3Error


class MinioBackend:
    def __init__(
        self,
        endpoint: str,
        access_key: str,
        secret_key: str,
        bucket: str,
        secure: bool = False,
    ):
        self.bucket = bucket
        self.client = Minio(
            endpoint,
            access_key=access_key,
            secret_key=secret_key,
            secure=secure,
        )

    def list_objects(self, prefix: str = "", recursive: bool = True):
        """
        Returns a generator of object names.
        """
        objects = self.client.list_objects(
            self.bucket,
            prefix=prefix,
            recursive=recursive,
        )
        for obj in objects:
            yield obj.object_name

    def get_object(self, key: str) -> bytes:
        response = self.client.get_object(self.bucket, key)
        data = response.read()
        response.close()
        response.release_conn()
        return data

    def bucket_exists(self) -> bool:
        return self.client.bucket_exists(self.bucket)


## Class for testing behaviour of MinIO backend
def main():
    # Adjust to your MinIO config
    backend = MinioBackend(
        endpoint="localhost:9000",
        access_key="minioadmin",
        secret_key="minioadmin",
        bucket="jetson-reid",
        secure=False,
    )

    try:
        if not backend.bucket_exists():
            print(f"Bucket '{backend.bucket}' does not exist.")
            return

        print(f"Listing objects in bucket '{backend.bucket}':\n")

        count = 0
        for obj_name in backend.list_objects():
            print(obj_name)
            count += 1

        print(f"\nTotal objects: {count}")

    except S3Error as e:
        print("S3 error:", e)


if __name__ == "__main__":
    main()