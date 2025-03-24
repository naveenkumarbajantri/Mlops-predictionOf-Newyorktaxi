pipeline {
    agent any

    stages {
        stage('Install Dependencies') {
            steps {
                sh 'pip install -r requirements.txt'
            }
        }

        stage('Train Model') {
            steps {
                dir('train') {
                    sh 'python train.py'
                }
            }
        }

        stage('Start API Server') {
            steps {
                dir('serve') {
                    // run serve.py in the background
                    sh 'nohup python serve.py &'
                }
            }
        }

        stage('Test Prediction') {
            steps {
                sh 'sleep 10'  // wait for serve.py to start
                sh 'python test-requirements.py'
            }
        }
    }
}
