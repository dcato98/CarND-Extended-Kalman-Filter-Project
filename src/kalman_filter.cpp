#include "kalman_filter.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;

KalmanFilter::KalmanFilter() {}

KalmanFilter::~KalmanFilter() {}

void KalmanFilter::Init(VectorXd &x_in, MatrixXd &P_in, MatrixXd &F_in,
                        MatrixXd &H_in, MatrixXd &R_in, MatrixXd &Q_in) {
  x_ = x_in;
  P_ = P_in;
  F_ = F_in;
  H_ = H_in;
  R_ = R_in;
  Q_ = Q_in;
  I_ = MatrixXd::Identity(x_.size(), x_.size());
}

void KalmanFilter::Predict() {
  /**
    * predict the state
  */
  x_ = F_*x_;
  P_ = F_*P_*F_.transpose() + Q_;
}

void KalmanFilter::Update(const VectorXd &z) {
  /**
    * update the state by using Kalman Filter equations
  */
  MatrixXd Ht = H_.transpose();
  VectorXd y = z - H_*x_;
  MatrixXd S = H_*P_*Ht + R_;
  MatrixXd K = P_*Ht*S.inverse();
  x_ += K*y;
  P_ = (I_-K*H_)*P_;
}

void KalmanFilter::UpdateEKF(const VectorXd &z) {
  /**
    * update the state by using Extended Kalman Filter equations
  */
  float px = x_[0];
  float py = x_[1];
  float vx = x_[2];
  float vy = x_[3];
  VectorXd hx(3);
  hx << sqrt(px*px+py*py), atan2(py,px), (px*vx+py*vy)/sqrt(px*px+py*py);
  MatrixXd Ht = H_.transpose();

  VectorXd y = z - hx;
  // nomralize the angle, phi, in y vector
  float pi = 3.14159265;
  while (y[1] < -pi){
    y[1] += 2*pi;
  }
  while (y[1] > pi){
    y[1] -= 2*pi;
  }
  MatrixXd S = H_*P_*Ht + R_;
  MatrixXd K = P_*Ht*S.inverse();
  x_ += K*y;
  P_ = (I_-K*H_)*P_;
}
