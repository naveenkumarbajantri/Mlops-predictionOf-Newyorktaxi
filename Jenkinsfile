pipeline {
    agent any

    stages {
        stage('Install Dependencies') {
            steps {
                bat 'pip install -r train/requirements.txt || pip install mlflow pandas scikit-learn flask requests'
            }
        }

        stage('Train Model') {
            steps {
                dir('train') {
                    bat 'python train.py'
                }
            }
        }

        stage('Start API Server') {
            steps {
                dir('serve') {
                    bat 'start /B python serve.py'
                }
            }
        }

        stage('Test Prediction') {
            steps {
                bat 'timeout /t 5'
                bat 'python test-requirements.py'
            }
        }
    }
}
