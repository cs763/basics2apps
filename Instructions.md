# Instruction for accessing Amazon EC2 Instances

## Instances Details
https://docs.google.com/spreadsheets/d/1VY7HkTDS4G2I-ar2eSa79lE-fDxUIvQ20s4TOKn350I/edit?usp=sharing

## Download the key file from here
https://www.cse.iitb.ac.in/~safeer/DLWorkshop.pem

## Key Permission override
chmod 400 DLWorkshop.pem

## Tunneling & SSH
ssh -i DLWorkshop.pem -L 8888:127.0.0.1:8888 ubuntu@ec2-###-##-##-###.compute-1.amazonaws.com


## SSH Only
ssh -i DLWorkshop.pem ubuntu@ec2-###-##-##-###.ap-south-1.compute.amazonaws.com
