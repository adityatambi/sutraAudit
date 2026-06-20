import uuid

class Datasets:
    def __init__(self, client):
        self.client = client

    def upload_file(self, file_path):
        """Mock file upload to Adaption Labs server."""
        print(f"\n[Adaption SDK] Mock: Ingesting local file '{file_path}'...")
        dataset_id = f"ds_{uuid.uuid4().hex[:8]}"
        print(f"[Adaption SDK] Mock: Generated dataset ID: {dataset_id}")
        return {"dataset_id": dataset_id, "status": "uploaded"}

    def run(self, dataset_id, column_mapping, brand_controls=None, job_specification=None):
        """Mock running a dataset adaptation job with Blueprint controls."""
        print(f"\n[Adaption SDK] Mock: Triggering dataset job...")
        print(f"[Adaption SDK] Mock: Target Dataset ID: {dataset_id}")
        print(f"[Adaption SDK] Mock: Column Mapping: {column_mapping}")
        
        has_blueprint = False
        if brand_controls:
            print(f"[Adaption SDK] Mock: Brand Controls:")
            for k, v in brand_controls.items():
                if k == "blueprint":
                    has_blueprint = True
                    print(f"  - {k}: \"{v[:60]}...\"")
                else:
                    print(f"  - {k}: {v}")
                    
        run_id = f"run_{uuid.uuid4().hex[:8]}"
        print(f"[Adaption SDK] Mock: Submitted job. Run ID: {run_id}")
        return Run(run_id, dataset_id, brand_controls, has_blueprint)


class Run:
    def __init__(self, run_id, dataset_id, brand_controls, has_blueprint):
        self.run_id = run_id
        self.dataset_id = dataset_id
        self.brand_controls = brand_controls
        self.has_blueprint = has_blueprint
        self.status = "COMPLETED"

    def get_status(self):
        """Mock retrieving task/run status."""
        return self.status

    def export_results(self):
        """Mock exporting/downloading the aligned dataset."""
        print(f"\n[Adaption SDK] Mock: Exporting aligned dataset for Run {self.run_id}...")
        return {
            "run_id": self.run_id,
            "has_blueprint": self.has_blueprint,
            "status": "success"
        }


class Adaption:
    def __init__(self, api_key=None):
        """Mock Adaption SDK client initialization."""
        self.api_key = api_key or "mock_pt_live_sutra_audit_dev_key"
        masked_key = self.api_key[:8] + "..." + self.api_key[-4:] if len(self.api_key) > 12 else self.api_key
        print(f"[Adaption SDK] Mock: Initialized Adaption client with API Key: {masked_key}")
        self.datasets = Datasets(self)
