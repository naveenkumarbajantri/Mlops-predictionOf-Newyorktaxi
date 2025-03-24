pipeline {
    agent any

    environment {
        PYTHON = '"C:\\Users\\Naveen kumar\\AppData\\Local\\Programs\\Python\\Python39\\python.exe"'
    }

    stages {
        stage('Install Dependencies') {
            steps {
                bat "${PYTHON} -m pip install --upgrade pip"
                bat "${PYTHON} -m pip install -r train\\requirements.txt"
                bat "${PYTHON} -m pip install flask requests mlflow scikit-learn pandas"
            }
        }

        stage('Train Model') {
            steps {
                bat "cd train && ${PYTHON} train.py"
            }
        }

        stage('Start Server') {
            steps {
                bat "start \"\" ${PYTHON} serve.py"
                sleep(time: 10, unit: 'SECONDS')
            }
        }

        stage('Run Prediction') {
            steps {
                bat "${PYTHON} test-requirements.py"
            }
        }
    }

    post {
        always {
            echo '✅ CI/CD/CT pipeline finished.'
        }
    }
}
