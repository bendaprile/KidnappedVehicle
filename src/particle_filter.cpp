/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>
#include <limits>

#include "helper_functions.h"

using std::default_random_engine;
using std::string;
using std::vector;
using std::normal_distribution;
using std::numeric_limits;
using std::cout;
using std::endl;
using std::uniform_int_distribution;
using std::uniform_real_distribution;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  /**
   * TODO: Set the number of particles. Initialize all particles to 
   *   first position (based on estimates of x, y, theta and their uncertainties
   *   from GPS) and all weights to 1. 
   * TODO: Add random Gaussian noise to each particle.
   * NOTE: Consult particle_filter.h for more information about this method 
   *   (and others in this file).
   */
  num_particles = 50;  // TODO: Set the number of particles
  
  default_random_engine gen;
  
  // Set sensor noise gaussian distributions
  normal_distribution<double> x_dist(x, std[0]);
  normal_distribution<double> y_dist(y, std[1]);
  normal_distribution<double> head_dist(theta, std[2]);
  
  // Initialize particles using gps location with added noise
  for(int i=0; i<num_particles; i++) {
    Particle newParticle;
    newParticle.id = i;
    newParticle.x = x_dist(gen);
    newParticle.y = y_dist(gen);
    newParticle.theta = head_dist(gen);
    newParticle.weight = 1.0;
    
    particles.push_back(newParticle);
  }
  
  is_initialized = true;

}

void ParticleFilter::prediction(double delta_t, double std_pos[], 
                                double velocity, double yaw_rate) {
  /**
   * TODO: Add measurements to each particle and add random Gaussian noise.
   * NOTE: When adding noise you may find std::normal_distribution 
   *   and std::default_random_engine useful.
   *  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
   *  http://www.cplusplus.com/reference/random/default_random_engine/
   */
  
  default_random_engine gen;
  
  for(int i=0; i<num_particles; i++) {
    
    double x_0 = particles[i].x;
    double y_0 = particles[i].y;
    double theta_0 = particles[i].theta;
    
    // Set sensor noise gaussian distributions
  	normal_distribution<double> x_dist(0, std_pos[0]);
  	normal_distribution<double> y_dist(0, std_pos[1]);
  	normal_distribution<double> theta_dist(0, std_pos[2]);
    
    // If heading is zero use the equations for 0 yaw rate
    if (abs(yaw_rate) < 0.0001) {
      
      // Define reused variable, change in velocity over timestep delta_t
      double velocityChange = velocity * delta_t;
      
      particles[i].x = x_0 + velocityChange * cos(theta_0);
      particles[i].y = y_0 + velocityChange * sin(theta_0);
    } else {
      // Define reused variables velocity/theta and yaw_rate * delta_t
      double velocity_yawrate = velocity / yaw_rate;
      double yawrate_dt = yaw_rate * delta_t;
      
      particles[i].x += x_0 + velocity_yawrate * (sin(theta_0 + yawrate_dt) - sin(theta_0));
      particles[i].y += y_0 + velocity_yawrate * (cos(theta_0) - cos(theta_0 + yawrate_dt));
      particles[i].theta = theta_0 + yawrate_dt;
    }
    
    // Add noises generated above
    particles[i].x += x_dist(gen);
    particles[i].y += y_dist(gen);
    particles[i].theta += theta_dist(gen);
    
//     cout << "x: " << particles[i].x << endl;
//     cout << "y: " << particles[i].y << endl;
//     cout << "theta: " << particles[i].theta << endl;
  }
}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted, 
                                     vector<LandmarkObs>& observations) {
  /**
   * TODO: Find the predicted measurement that is closest to each 
   *   observed measurement and assign the observed measurement to this 
   *   particular landmark.
   * NOTE: this method will NOT be called by the grading code. But you will 
   *   probably find it useful to implement this method and use it as a helper 
   *   during the updateWeights phase.
   */
  
  for(unsigned int i=0; i<observations.size(); i++) {
    double minDist = numeric_limits<double>::max();
    LandmarkObs observation = observations[i];
    int landmark_index;
    
    for(unsigned int j=0; j<predicted.size(); j++) {
      LandmarkObs prediction = predicted[j];
      
      double currentDist = dist(observation.x, observation.y, prediction.x, prediction.y);
      
      if(currentDist < minDist) {
        minDist = currentDist;
        landmark_index = prediction.id;
      }
    }
    
    // Assign observations id to the id of the closest landmark
    observations[i].id = landmark_index;
  }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
                                   const vector<LandmarkObs> &observations, 
                                   const Map &map_landmarks) {
  /**
   * TODO: Update the weights of each particle using a mult-variate Gaussian 
   *   distribution. You can read more about this distribution here: 
   *   https://en.wikipedia.org/wiki/Multivariate_normal_distribution
   * NOTE: The observations are given in the VEHICLE'S coordinate system. 
   *   Your particles are located according to the MAP'S coordinate system. 
   *   You will need to transform between the two systems. Keep in mind that
   *   this transformation requires both rotation AND translation (but no scaling).
   *   The following is a good resource for the theory:
   *   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
   *   and the following is a good resource for the actual equation to implement
   *   (look at equation 3.33) http://planning.cs.uiuc.edu/node99.html
   */
  
  // Calculate the gaussian normalizer for the multivariate probability equation
  double weight_normalizer = 0.0;
  
  for(int i=0; i<num_particles; i++) {
    vector<LandmarkObs> transformedObs;
    
    double xPart = particles[i].x;
    double yPart = particles[i].y;
    double theta = particles[i].theta;
    
    // Transforming from vehicle coordinates to map coordinates
    for(unsigned int j=0; j<observations.size(); j++) {
      double x_obs = observations[j].x;
      double y_obs = observations[j].y;
      
      LandmarkObs observation;
      
      // Transformations for x and y
      observation.x = xPart + (cos(theta) * x_obs) - (sin(theta) * y_obs);
      observation.y = yPart + (sin(theta) * x_obs) + (cos(theta) * y_obs);
      
      transformedObs.push_back(observation);
    }
    
    vector<LandmarkObs> predictions;
      
    // Generating predictions vector with particles that are within sensor range
    for(unsigned int j=0; j<map_landmarks.landmark_list.size(); j++) {
      LandmarkObs inRange;
      double xLandmark = map_landmarks.landmark_list[j].x_f;
      double yLandmark = map_landmarks.landmark_list[j].y_f;
      double idLandmark = map_landmarks.landmark_list[j].id_i;
        
      if(dist(xLandmark, yLandmark, xPart, yPart) <= sensor_range){
        //add to prediction vector
        inRange.x = xLandmark;
        inRange.y = yLandmark;
        inRange.id = idLandmark;
        predictions.push_back(inRange);
      }
    }
    
    // Associate closest landmark to the current particle and reset weight to 1.0
    dataAssociation(predictions, transformedObs);
    particles[i].weight = 1.0;
    
    for(unsigned int j=0; j<transformedObs.size(); j++) {
      double x_obs = transformedObs[j].x;
      double y_obs = transformedObs[j].y;
      double nearestLandmark = transformedObs[j].id;
      
      for(unsigned int k=0; k<predictions.size(); k++) {
        if(predictions[k].id == nearestLandmark) {
          double mu_x = predictions[k].x;
          double mu_y = predictions[k].y;
          //cout << "Particles Size: " << particles.size() << endl;
          //cout << "Predictions Size: " << predictions.size() << endl;

          // Compute updated weight with multiv_prob function
          double weightObs = multiv_prob(std_landmark[0], std_landmark[1], x_obs, y_obs, mu_x, mu_y);
          particles[i].weight *= weightObs;
        } else {
          //cout << "Particle did not match closest landmark" << endl;
          //cout << "Pred id: " << predictions[k].id << endl;
          //cout << "Nearest Landmark: " << nearestLandmark << endl;
        }
      }
    }
    weight_normalizer += particles[i].weight;
  }
  
  cout << "weight_normalizer: " << weight_normalizer << endl;
  
  // Normalize the weights
  for (unsigned int i = 0; i < particles.size(); i++) {
      particles[i].weight /= weight_normalizer;
//     cout << "weights i: " << weights[i] << endl;
//     cout << "particles weight i: " << particles[i].weight << endl;
  }
}

void ParticleFilter::resample() {
  /**
   * TODO: Resample particles with replacement with probability proportional 
   *   to their weight. 
   * NOTE: You may find std::discrete_distribution helpful here.
   *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */
  
  default_random_engine gen;
  vector<Particle> newParticles;
  
  vector<double> weights;
  for (int i = 0; i < num_particles; i++) {
    weights.push_back(particles[i].weight);
  }
  
  // Grab a random index to start the resampling from
  uniform_int_distribution<int> indexDist(0, num_particles-1);
  int index = indexDist(gen);
  
  // Grab the maximum weight in the weights vector
  double maxWeight = *max_element(weights.begin(), weights.end());
  
  double beta = 0.0;
  
  uniform_real_distribution<double> indexRealDist(0.0, maxWeight);
  
  // Resample the particles
  for (int i = 0; i < num_particles; i++) {
    beta += indexRealDist(gen) * 2.0;
    
    while (beta > weights[index]) {
      beta -= weights[index];
      index = (index + 1) % num_particles;
    }
    newParticles.push_back(particles[index]);
  }
  particles = newParticles;
}

// Exact function was used in the Implementation of a Particle Filter lesson
double ParticleFilter::multiv_prob(double sig_x, double sig_y, double x_obs, double y_obs,
                   double mu_x, double mu_y) {
  // calculate normalization term
  double gauss_norm;
  gauss_norm = 1 / (2 * M_PI * sig_x * sig_y);

  // calculate exponent
  double exponent;
  exponent = (pow(x_obs - mu_x, 2) / (2 * pow(sig_x, 2)))
               + (pow(y_obs - mu_y, 2) / (2 * pow(sig_y, 2)));
    
  // calculate weight using normalization terms and exponent
  double weight;
  weight = gauss_norm * exp(-exponent);
    
  return weight;
}

void ParticleFilter::SetAssociations(Particle& particle, 
                                     const vector<int>& associations, 
                                     const vector<double>& sense_x, 
                                     const vector<double>& sense_y) {
  // particle: the particle to which assign each listed association, 
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
  vector<double> v;

  if (coord == "X") {
    v = best.sense_x;
  } else {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}