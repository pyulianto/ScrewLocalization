# Git Branch Strategy

## Branch Structure

### Main Branches

- **`main`** - Production-ready code. Always stable and deployable.
- **`develop`** - Integration branch for ongoing development. All feature branches merge here first.

### Feature Branches

Feature branches follow the implementation phases:

- **`feature/phase1-analysis`** - Analysis and preparation phase
  - Collect sample data
  - Analyze vector patterns
  - Define detection rules

- **`feature/phase2-head-detection`** - Enhanced head detection
  - Polygon detection
  - Composite head detection
  - Integration with existing circle detection

- **`feature/phase3-thread-detection`** - Enhanced thread detection
  - Saw-tooth pattern detection
  - Zig-zag pattern detection
  - Radial thread detection

- **`feature/phase4-composite-matching`** - Composite shape matching
  - Component matching logic
  - Shape validation
  - Complete screw detection

- **`feature/phase5-integration`** - Integration and enhancement
  - Update main detection pipeline
  - Enhance classification
  - Improve confidence scoring

- **`feature/phase6-testing`** - Testing and validation
  - Test suite creation
  - Validation on real PDFs
  - Iteration and refinement

## Workflow

### Starting Work on a Phase

1. **Checkout the feature branch:**
   ```bash
   git checkout feature/phase2-head-detection
   ```

2. **Make your changes:**
   - Implement the phase features
   - Test locally
   - Commit frequently with descriptive messages

3. **Push to remote:**
   ```bash
   git push origin feature/phase2-head-detection
   ```

### Merging a Feature Branch

1. **Switch to develop:**
   ```bash
   git checkout develop
   git pull origin develop
   ```

2. **Merge feature branch:**
   ```bash
   git merge feature/phase2-head-detection
   ```

3. **Push to remote:**
   ```bash
   git push origin develop
   ```

4. **After testing, merge to main:**
   ```bash
   git checkout main
   git merge develop
   git push origin main
   ```

### Branch Naming Convention

- Feature branches: `feature/phaseX-description`
- Bug fixes: `fix/description`
- Hotfixes: `hotfix/description`

## Current Branch Status

All feature branches are created and ready for development:
- ✅ `develop` - Main development branch
- ✅ `feature/phase1-analysis`
- ✅ `feature/phase2-head-detection`
- ✅ `feature/phase3-thread-detection`
- ✅ `feature/phase4-composite-matching`
- ✅ `feature/phase5-integration`
- ✅ `feature/phase6-testing`

## Best Practices

1. **Always start from develop** when creating new feature branches
2. **Commit frequently** with clear, descriptive messages
3. **Test before merging** to develop
4. **Keep branches up to date** with develop
5. **Delete merged branches** after merging to main

## Example Workflow

```bash
# Start Phase 2 work
git checkout develop
git pull origin develop
git checkout -b feature/phase2-head-detection

# Make changes and commit
git add .
git commit -m "Add polygon head detection"

# Push to remote
git push origin feature/phase2-head-detection

# When complete, merge to develop
git checkout develop
git merge feature/phase2-head-detection
git push origin develop
```

