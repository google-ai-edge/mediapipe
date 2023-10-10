pipeline {
    agent {
      label 'ovmscheck'
    }
    stages {
        stage('Configure') {
          steps {
            script {
              checkout scm
              shortCommit = sh(returnStdout: true, script: "git log -n 1 --pretty=format:'%h'").trim()
              echo shortCommit
            }
          }     
        }   
        stage("build image") {
          steps {
              sh script: "make docker_build OVMS_MEDIA_IMAGE_TAG=${shortCommit}"
          }    
        }
        stage("unit tests") {
          steps {
              sh script: "make tests OVMS_MEDIA_IMAGE_TAG=${shortCommit}"
          }
        }
        stage("test demos") {
          steps {
              sh script: "make run_demos_in_docker OVMS_MEDIA_IMAGE_TAG=${shortCommit}"
          }
        }
    }
}

