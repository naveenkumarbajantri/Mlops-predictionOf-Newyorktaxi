pipeline {
    agent any

    environment {
        PYTHON_PATH = '"C:\\Users\\Naveen kumar\\AppData\\Local\\Programs\\Python\\Python39"'
    }

    stages {
        stage('Install Dependencies') {
            steps {
                bat '''
                %PYTHON_PATH%\\python.exe -m pip install --upgrade pip
                %PYTHON_PATH%\\python.exe -m pip install -r train/requirements.txt
                %PYTHON_PATH%\\python.exe -m pip install mlflow pandas scikit-learn flask requests
                '''
            }
        }

        stage('Train Model') {
            steps {
                bat '''
                cd train
                %PYTHON_PATH%\\python.exe train.py
                '''
            }
        }

        stage('Run Prediction') {
            steps {
                bat '''
                %PYTHON_PATH%\\python.exe test-requirements.py
                '''
            }
        }
    }

    post {
        always {
            echo 'CI/CD/CT pipeline finished.'
        }
    }
}
