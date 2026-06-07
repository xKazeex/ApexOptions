from fastapi import APIRouter
from typing import List
from api.models.schemas import AnalysisResponse, ApexStrategyReport, ReportMetadata
import subprocess
import json

router = APIRouter()

@router.get("/history", response_model=List[AnalysisResponse])
async def get_history():
    try:
        sql = "SELECT report_json, metadata_json FROM analysis_history ORDER BY timestamp DESC LIMIT 20"
        result = subprocess.run(["team-db", sql], capture_output=True, text=True)
        if result.returncode == 0:
            rows = json.loads(result.stdout)
            history = []
            for row in rows:
                report = ApexStrategyReport.model_validate_json(row['report_json'])
                metadata = ReportMetadata.model_validate_json(row['metadata_json'])
                history.append(AnalysisResponse(report=report, metadata=metadata))
            return history
        return []
    except Exception as e:
        print(f"Error reading history: {e}")
        return []
