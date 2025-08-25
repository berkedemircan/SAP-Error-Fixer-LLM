import json


with open("SAP_Logs_with_Solutions.json", "r", encoding="utf-8") as f:
    logs = json.load(f)

with open("Sap_Logs.txt", "w", encoding="utf-8") as out:
    for entry in logs:
        job = entry.get("job_name", "")
        typ = entry.get("type", "")
        err = entry.get("error", "")
        en = entry.get("solution_en", "")
        tr = entry.get("solution_tr", "")
        out.write(f"Job: {job}\nType: {typ}\nError: {err}\nSolution EN: {en}\nSolution TR: {tr}\n---\n")

#Converting json data file to txt file