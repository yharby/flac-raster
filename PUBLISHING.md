# Publishing to PyPI

This document describes how to publish FLAC-Raster to PyPI using GitHub Actions.

## Setup

### 1. PyPI Account Setup

1. Create an account on [PyPI](https://pypi.org/)
2. Enable 2FA on your account
3. Create an API token:
   - Go to Account Settings → API tokens
   - Create a token with scope "Entire account" or specific to this project
   - Copy the token (starts with `pypi-`)

### 2. GitHub Repository Setup

1. Go to your repository Settings → Secrets and variables → Actions
2. Add a new repository secret:
   - Name: `PYPI_API_TOKEN`
   - Value: Your PyPI API token (including the `pypi-` prefix)

### 3. Set up Trusted Publishing (Recommended)

For enhanced security, you can use PyPI's trusted publishing instead of API tokens:

1. Go to [PyPI Trusted Publishers](https://pypi.org/manage/account/publishing/)
2. Add a new trusted publisher:
   - Repository name: `yharby/flac-raster`
   - Workflow name: `ci.yml`
   - Environment name: `release`

## Publishing Process

### Automatic Publishing (Recommended)

The GitHub Actions workflow will automatically publish to PyPI when you create a release:

1. **Update version** in `pyproject.toml`
2. **Create a git tag**:
   ```bash
   git tag v0.1.0
   git push origin v0.1.0
   ```
3. **Create a GitHub release** from the tag
4. The CI workflow will automatically:
   - Run tests on multiple platforms
   - Build the package
   - Publish to PyPI

### Manual Publishing

If you need to publish manually:

```bash
# Install build tools
pip install build twine

# Build the package
python -m build

# Check the package
python -m twine check dist/*

# Upload to PyPI (test first)
python -m twine upload --repository testpypi dist/*

# Upload to PyPI (production)
python -m twine upload dist/*
```

## Version Management

This project uses semantic versioning (SemVer):

- `MAJOR.MINOR.PATCH` (e.g., `1.2.3`)
- Increment MAJOR for breaking changes
- Increment MINOR for new features
- Increment PATCH for bug fixes

Update the version in `pyproject.toml`:

```toml
[project]
version = "0.1.0"
```

## Release Checklist

Before creating a release:

- [ ] Update version in `pyproject.toml`
- [ ] Update `CHANGELOG.md` (if exists)
- [ ] Run tests locally: `pytest tests/`
- [ ] Build and test package: `python -m build && python -m twine check dist/*`
- [ ] Create git tag and push
- [ ] Create GitHub release
- [ ] Verify CI passes
- [ ] Check PyPI upload

## Troubleshooting

### Common Issues

1. **Package already exists**: You can't overwrite a version on PyPI. Increment the version number.

2. **Authentication failed**: Check your API token in GitHub secrets.

3. **Build failed**: Ensure all dependencies are correctly specified in `pyproject.toml`.

4. **Tests failed**: Fix test failures before releasing.

### Testing the Package

After publishing, test the installation:

```bash
# Create a new virtual environment
python -m venv test_env
source test_env/bin/activate  # On Windows: test_env\Scripts\activate

# Install from PyPI
pip install flac-raster

# Test the CLI
flac-raster --help
```

## GitHub Actions Configuration

The CI workflow (`.github/workflows/ci.yml`) handles:

- Testing on multiple Python versions (3.9-3.12)
- Testing on multiple OS (Ubuntu, Windows, macOS)
- Building the package
- Publishing to PyPI on release

Key environment variables:

- `PYPI_API_TOKEN`: Your PyPI API token (set in GitHub secrets)

## Security Notes

- Never commit API tokens to the repository
- Use GitHub's encrypted secrets for sensitive data
- Consider using PyPI's trusted publishing for enhanced security
- Regularly rotate your PyPI API tokens
