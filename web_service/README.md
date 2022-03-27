# Steps for setting up a web service 

1. Create open security group

allow rules for all https traffic incoming and outgoing

2. Create your EC2 instance
Ubuntu Server 18.04 LTS (HVM), SSD Volume Type
t2 micro


3. Upload your material  

(example --> you'll replace with public IP and your PEM)
ssh -i tinghaole_public.pem ubuntu@34.217.116.223 -v
scp -i tinghaole_public.pem -r web_service/ ubuntu@34.217.116.223:~



4. spin up with docker
sudo apt-get update
sudo curl -L "https://github.com/docker/compose/releases/download/1.23.2/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose
sudo ln -s /usr/local/bin/docker-compose /usr/bin/docker-compose

--> not sudo docker compose will cause these things to be out of date


5. Create target group
name something like tinghaole-group
note the availibity zone 
assign VM from above to this availibitly group

6. create load balancer
set availibty zone inclusive of the target group
set open security group to allow HTTPS traffic


7. create certificate
register certificate to DNS to predictor.tinghaole.com

check that load balancer returns rational answers 
hit it here http://tinghaole-balancer-1948235521.us-west-2.elb.amazonaws.com/predict


8. Update domain endpoint
set routing from predictor.tinghaole.com to the address of the load balancer.