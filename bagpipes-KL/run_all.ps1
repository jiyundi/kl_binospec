for ($i = 1; $i -le 142; $i++) {
    Write-Host "Running slitID $i ..."
    try {
        python fitting_spec_by_Bagpipes.py --slitID $i
    } catch {
        Write-Host "Error on slitID $i, skipping..."
    }
}
