# Prompt Management System

This is a password-protected web application for managing prompts used in the VQA system.

## Features

- Password protection for secure access
- Edit prompts with placeholder preservation
- Version history for all changes
- Ability to revert to previous versions
- Updates the main prompts.py file automatically

## Setup and Running

1. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Run the Streamlit application:
   ```
   streamlit run app.py
   ```

3. Access the application in your browser at http://localhost:8501

## Default Login

- Default password: `admin123`
- You can change the password after logging in

## Usage

1. Log in with your credentials
2. Select a prompt to edit from the sidebar
3. Make your changes while preserving all placeholders
4. Save your changes
5. Click "Update prompts.py" to apply changes to the main file
6. View and restore previous versions if needed

## Important Notes

- All placeholders in the format `{placeholder_name}` must be preserved
- Each change is recorded with timestamp and username
- You can revert to any previous version at any time
