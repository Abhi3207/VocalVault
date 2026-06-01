<#
.SYNOPSIS
    Daily auto-push script for VocalVault.
    Creates/updates a branch named ddmmyyyy and pushes all changes.

.DESCRIPTION
    - Checks for uncommitted changes (staged, unstaged, untracked)
    - Creates or switches to a branch named after today's date (ddmmyyyy)
    - Stages all changes, commits with a timestamp, and pushes to origin
    - Logs output to auto_push.log in the repo root
#>

$ErrorActionPreference = "Continue"

# --- Configuration ---
$repoPath = "c:\Users\abhin\OneDrive\Documents\my_projects\audio-rag"
$logFile  = Join-Path $repoPath "auto_push.log"

# --- Helper: Write to log and console ---
function Write-Log {
    param([string]$Message)
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $entry = "[$timestamp] $Message"
    Write-Host $entry
    Add-Content -Path $logFile -Value $entry
}

function Invoke-Git {
    param([string]$Args_)
    $output = & cmd /c "git $Args_ 2>&1"
    $exitCode = $LASTEXITCODE
    $outputStr = ($output | Out-String).Trim()
    if ($outputStr) { Write-Log $outputStr }
    if ($exitCode -ne 0) {
        throw "git $Args_ failed with exit code $exitCode"
    }
}

# --- Main ---
try {
    Set-Location $repoPath
    Write-Log "=== Auto-push started ==="

    # Check if there are any changes (staged, unstaged, or untracked)
    $status = git status --porcelain 2>&1
    if ([string]::IsNullOrWhiteSpace($status)) {
        Write-Log "No changes detected. Skipping push."
        Write-Log "=== Auto-push finished (no-op) ==="
        exit 0
    }

    Write-Log "Changes detected:"
    Write-Log ($status | Out-String)

    # Branch name: ddmmyyyy
    $branchName = Get-Date -Format "ddMMyyyy"
    Write-Log "Target branch: $branchName"

    # Stash uncommitted changes so branch switching doesn't fail
    Write-Log "Stashing local changes..."
    Invoke-Git "stash --include-untracked"

    # Check if the branch already exists locally
    $branchExists = git branch --list $branchName 2>&1
    if ([string]::IsNullOrWhiteSpace($branchExists)) {
        # Branch doesn't exist — create it from main
        Write-Log "Creating new branch '$branchName' from 'main'..."
        Invoke-Git "checkout main"
        Invoke-Git "pull origin main"
        Invoke-Git "checkout -b $branchName"
    }
    else {
        # Branch exists — switch to it
        Write-Log "Branch '$branchName' already exists. Switching to it..."
        Invoke-Git "checkout $branchName"
    }

    # Restore stashed changes
    Write-Log "Restoring stashed changes..."
    Invoke-Git "stash pop"

    # Stage all changes
    Invoke-Git "add -A"

    # Commit with timestamp
    $commitMsg = "auto-push: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
    Write-Log "Committing: $commitMsg"
    Invoke-Git "commit -m `"$commitMsg`""

    # Push to origin
    Write-Log "Pushing to origin/$branchName..."
    Invoke-Git "push origin $branchName"

    Write-Log "=== Auto-push completed successfully ==="
}
catch {
    Write-Log "ERROR: $_"
    Write-Log "=== Auto-push FAILED ==="
    exit 1
}
