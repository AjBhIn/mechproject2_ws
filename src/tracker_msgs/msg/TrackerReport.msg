# The opponent's photo
sensor_msgs/CompressedImage image

# My own ID (e.g., 1 if you are tb401)
uint8 my_id

# My opponent ID (e.g., 9 if your enemy is tb409)
uint8 opp_id

# Normed distance of your opponent w.r.t. your frame
float32 rel_dist

# Orientation (yaw) of your opponent w.r.t. your frame in degrees
float32 rel_orient