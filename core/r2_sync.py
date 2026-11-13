import io
import json
import logging
import os
import tempfile
import threading
from datetime import datetime, timezone
from typing import Any, Dict, List, Mapping, Optional

logger = logging.getLogger("mips.r2_sync")


class FakeS3Client:
    """In-memory S3-compatible client for deterministic unit testing."""

    def __init__(self, max_keys: int = 1000) -> None:
        self.objects: Dict[str, bytes] = {}
        self.max_keys = max_keys
        self.get_count = 0
        self.put_count = 0

    def list_objects_v2(
        self,
        Bucket: str,
        Prefix: str = "",
        ContinuationToken: Optional[str] = None,
        MaxKeys: Optional[int] = None,
    ) -> Dict[str, Any]:
        limit = MaxKeys or self.max_keys
        keys = sorted(k for k in self.objects if k.startswith(Prefix))
        start = int(ContinuationToken) if ContinuationToken is not None else 0
        page = keys[start : start + limit]
        next_idx = start + limit
        is_truncated = next_idx < len(keys)
        result: Dict[str, Any] = {
            "Contents": [{"Key": k, "Size": len(self.objects[k])} for k in page],
            "IsTruncated": is_truncated,
        }
        if is_truncated:
            result["NextContinuationToken"] = str(next_idx)
        return result

    def put_object(
        self, Bucket: str, Key: str, Body: bytes, ContentType: str = "application/octet-stream"
    ) -> Dict[str, Any]:
        self.put_count += 1
        self.objects[Key] = Body if isinstance(Body, (bytes, bytearray)) else bytes(Body)
        return {}

    def get_object(self, Bucket: str, Key: str) -> Dict[str, Any]:
        self.get_count += 1
        if Key not in self.objects:
            raise KeyError(f"NoSuchKey: {Key}")
        return {"Body": io.BytesIO(self.objects[Key])}

    def delete_objects(self, Bucket: str, Delete: Dict[str, List[Dict[str, str]]]) -> Dict[str, Any]:
        for obj in Delete.get("Objects", []):
            self.objects.pop(obj["Key"], None)
        return {}


class R2SyncManager:
    """Synchronizes MIPS local data directory with a bucket-scoped Cloudflare R2 store."""

    REQUIRED_KEYS = ("endpoint_url", "bucket", "access_key_id", "secret_access_key")

    def __init__(
        self,
        s3_client: Optional[Any] = None,
        bucket: Optional[str] = None,
        local_data_dir: str = "data",
        status: Optional[str] = None,
        error: Optional[str] = None,
    ) -> None:
        self.s3_client = s3_client
        self.bucket = bucket
        self.local_data_dir = local_data_dir
        self._lock = threading.Lock()
        self.last_sync_at: Optional[str] = None
        self.last_sync_error: Optional[str] = error
        if status is not None:
            self.last_sync_status = status
        elif self.is_configured():
            self.last_sync_status = "ok"
        else:
            self.last_sync_status = "offline"

    @classmethod
    def from_secrets(
        cls, secrets: Mapping[str, Any], local_data_dir: str = "data"
    ) -> "R2SyncManager":
        """Build an R2SyncManager from Streamlit secrets or return a safe offline/degraded manager."""
        try:
            r2_cfg = secrets.get("r2") if hasattr(secrets, "get") else None
        except Exception:
            r2_cfg = None

        if not r2_cfg or not isinstance(r2_cfg, Mapping):
            return cls(local_data_dir=local_data_dir, status="offline")

        missing = [k for k in cls.REQUIRED_KEYS if not str(r2_cfg.get(k, "")).strip()]
        if missing:
            msg = f"Incomplete [r2] secrets configuration (missing: {', '.join(missing)})"
            logger.warning(msg)
            return cls(local_data_dir=local_data_dir, status="degraded", error=msg)

        try:
            import boto3

            client = boto3.client(
                "s3",
                endpoint_url=str(r2_cfg["endpoint_url"]).strip(),
                aws_access_key_id=str(r2_cfg["access_key_id"]).strip(),
                aws_secret_access_key=str(r2_cfg["secret_access_key"]).strip(),
                region_name="auto",
            )
            return cls(
                s3_client=client,
                bucket=str(r2_cfg["bucket"]).strip(),
                local_data_dir=local_data_dir,
                status="ok",
            )
        except Exception as exc:
            msg = cls._sanitize_error(exc)
            logger.warning("Failed to initialize R2 client: %s", msg)
            return cls(local_data_dir=local_data_dir, status="degraded", error=msg)

    def is_configured(self) -> bool:
        return self.s3_client is not None and bool(self.bucket)

    @staticmethod
    def _sanitize_error(exc: Exception) -> str:
        text = str(exc)
        for token in ("secret_access_key", "access_key_id", "Signature", "Authorization"):
            if token.lower() in text.lower():
                return f"{type(exc).__name__}: R2 request failed (details redacted)"
        return f"{type(exc).__name__}: {text[:180]}"

    @staticmethod
    def _atomic_write_bytes(dest_path: str, data: bytes) -> None:
        parent = os.path.dirname(dest_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(dir=parent or ".", prefix=".tmp_")
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(data)
            os.replace(tmp_path, dest_path)
        except Exception:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            raise

    def _list_objects(self, prefix: str) -> List[Dict[str, Any]]:
        if not self.is_configured():
            return []
        items: List[Dict[str, Any]] = []
        token: Optional[str] = None
        while True:
            kwargs: Dict[str, Any] = {"Bucket": self.bucket, "Prefix": prefix}
            if token is not None:
                kwargs["ContinuationToken"] = token
            resp = self.s3_client.list_objects_v2(**kwargs)
            items.extend(resp.get("Contents", []) or [])
            if not resp.get("IsTruncated"):
                break
            token = resp.get("NextContinuationToken")
        return items

    def _get_bytes(self, key: str) -> bytes:
        resp = self.s3_client.get_object(Bucket=self.bucket, Key=key)
        return resp["Body"].read()

    def _put_bytes(self, key: str, data: bytes, content_type: str = "application/octet-stream") -> bool:
        if not self.is_configured():
            return False
        last_exc: Optional[Exception] = None
        for _ in range(2):
            try:
                self.s3_client.put_object(
                    Bucket=self.bucket, Key=key, Body=data, ContentType=content_type
                )
                self.last_sync_status = "ok"
                self.last_sync_error = None
                return True
            except Exception as exc:
                last_exc = exc
        if last_exc is not None:
            self.last_sync_status = "degraded"
            self.last_sync_error = self._sanitize_error(last_exc)
            logger.warning("R2 put failed for %s: %s", key, self.last_sync_error)
        return False

    def push_file(self, remote_key: str, local_path: str) -> bool:
        if not self.is_configured() or not os.path.exists(local_path):
            return False
        with open(local_path, "rb") as f:
            data = f.read()
        ctype = "application/json" if remote_key.endswith(".json") else "application/octet-stream"
        return self._put_bytes(remote_key, data, content_type=ctype)

    def push_run(self, record: Dict[str, Any]) -> bool:
        run_id = record.get("id")
        if not self.is_configured() or not run_id:
            return False
        payload = json.dumps(record, indent=2).encode("utf-8")
        return self._put_bytes(f"runs/{run_id}.json", payload, content_type="application/json")

    def push_model_artifact(
        self, run_id: str, local_dir: str, subdir: Optional[str] = None
    ) -> bool:
        if not self.is_configured() or not os.path.isdir(local_dir):
            return False
        ok = True
        prefix = f"models/{run_id}/{subdir}" if subdir else f"models/{run_id}"
        for fname in os.listdir(local_dir):
            fpath = os.path.join(local_dir, fname)
            if os.path.isfile(fpath):
                if not self.push_file(f"{prefix}/{fname}", fpath):
                    ok = False
        return ok

    def delete_prefix(self, prefix: str) -> int:
        if not self.is_configured():
            return 0
        objs = self._list_objects(prefix)
        keys = [o["Key"] for o in objs]
        if not keys:
            return 0
        for i in range(0, len(keys), 1000):
            batch = [{"Key": k} for k in keys[i : i + 1000]]
            self.s3_client.delete_objects(Bucket=self.bucket, Delete={"Objects": batch})
        return len(keys)

    def pull_metadata(self, db_path: Optional[str] = None) -> Dict[str, int]:
        """Pull dataset configs, split manifests, Gemini cache, and merge unseen runs/*.json."""
        if not self.is_configured():
            return {"runs_added": 0, "files_updated": 0}
        target_db = db_path or os.path.join(self.local_data_dir, "leaderboard_history.json")
        with self._lock:
            try:
                files_updated = 0
                for prefix in ("datasets/", "gemini_cache/"):
                    for obj in self._list_objects(prefix):
                        key = obj["Key"]
                        is_ds_meta = prefix == "datasets/" and (
                            key.endswith("/config.json") or key.endswith("/split_manifest.json")
                        )
                        is_cache = prefix == "gemini_cache/" and key.endswith(".json")
                        if is_ds_meta or is_cache:
                            dest = os.path.join(self.local_data_dir, key)
                            self._atomic_write_bytes(dest, self._get_bytes(key))
                            files_updated += 1

                existing_runs: List[Dict[str, Any]] = []
                if os.path.exists(target_db):
                    try:
                        with open(target_db, "r", encoding="utf-8") as f:
                            loaded = json.load(f)
                            if isinstance(loaded, list):
                                existing_runs = loaded
                    except (json.JSONDecodeError, FileNotFoundError):
                        existing_runs = []

                known_ids = {r.get("id") for r in existing_runs if r.get("id")}
                runs_added = 0
                for obj in self._list_objects("runs/"):
                    key = obj["Key"]
                    if not key.endswith(".json"):
                        continue
                    run_id = os.path.basename(key)[:-5]
                    if run_id in known_ids:
                        continue
                    rec = json.loads(self._get_bytes(key).decode("utf-8"))
                    if isinstance(rec, dict):
                        rec_id = rec.get("id") or run_id
                        if rec_id not in known_ids:
                            existing_runs.append(rec)
                            known_ids.add(rec_id)
                            runs_added += 1

                if runs_added > 0 or not os.path.exists(target_db):
                    existing_runs.sort(key=lambda r: r.get("timestamp", ""))
                    self._atomic_write_bytes(
                        target_db, json.dumps(existing_runs, indent=2).encode("utf-8")
                    )

                self.last_sync_at = datetime.now(timezone.utc).isoformat()
                self.last_sync_status = "ok"
                self.last_sync_error = None
                return {"runs_added": runs_added, "files_updated": files_updated}
            except Exception as exc:
                self.last_sync_status = "degraded"
                self.last_sync_error = self._sanitize_error(exc)
                logger.warning("R2 pull_metadata failed: %s", self.last_sync_error)
                return {"runs_added": 0, "files_updated": 0}

    def pull_dataset_prefix(self, dataset_id: str, prefix: str) -> int:
        """Lazily pull datasets/<dataset_id>/<prefix>/ skipping matching sizes; prune stale test_dataset files."""
        if not self.is_configured():
            return 0
        clean_prefix = prefix.strip("/")
        remote_prefix = f"datasets/{dataset_id}/{clean_prefix}/"
        local_root = os.path.join(self.local_data_dir, "datasets", dataset_id, clean_prefix)
        with self._lock:
            try:
                objs = self._list_objects(remote_prefix)
                remote_rel_paths = set()
                downloaded = 0
                for obj in objs:
                    key = obj["Key"]
                    if key.endswith("/"):
                        continue
                    rel = key[len(remote_prefix) :]
                    remote_rel_paths.add(rel)
                    dest = os.path.join(local_root, rel)
                    if os.path.exists(dest) and os.path.getsize(dest) == obj.get("Size"):
                        continue
                    self._atomic_write_bytes(dest, self._get_bytes(key))
                    downloaded += 1

                if clean_prefix == "test_dataset" and os.path.isdir(local_root):
                    for root, _, files in os.walk(local_root):
                        for fname in files:
                            fpath = os.path.join(root, fname)
                            rel = os.path.relpath(fpath, local_root).replace(os.sep, "/")
                            if rel not in remote_rel_paths:
                                os.remove(fpath)
                return downloaded
            except Exception as exc:
                self.last_sync_status = "degraded"
                self.last_sync_error = self._sanitize_error(exc)
                logger.warning("R2 pull_dataset_prefix(%s, %s) failed: %s", dataset_id, prefix, self.last_sync_error)
                return 0

    def ensure_training_pack_local(self, dataset_id: str, stage: str) -> bool:
        """Download datasets/<dataset_id>/training_packs/<stage>_train.zip if missing locally."""
        local_zip = os.path.join(
            self.local_data_dir, "datasets", dataset_id, "training_packs", f"{stage}_train.zip"
        )
        if os.path.exists(local_zip):
            return True
        if not self.is_configured():
            return False
        remote_key = f"datasets/{dataset_id}/training_packs/{stage}_train.zip"
        try:
            data = self._get_bytes(remote_key)
            self._atomic_write_bytes(local_zip, data)
            return True
        except Exception:
            return False

    def ensure_model_artifact_local(
        self, run_id: str, local_models_dir: str, subdir: Optional[str] = None
    ) -> bool:
        """Download models/<run_id>/[subdir/]submission.zip on demand if missing locally."""
        target_dir = os.path.join(local_models_dir, run_id, subdir) if subdir else os.path.join(local_models_dir, run_id)
        local_zip = os.path.join(target_dir, "submission.zip")
        if os.path.exists(local_zip):
            return True
        if not self.is_configured():
            return False
        remote_key = f"models/{run_id}/{subdir}/submission.zip" if subdir else f"models/{run_id}/submission.zip"
        try:
            data = self._get_bytes(remote_key)
            self._atomic_write_bytes(local_zip, data)
            return True
        except Exception:
            return False
