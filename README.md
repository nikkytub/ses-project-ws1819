# Smart Energy Systems Project WS-18/19
#### To import the database:
1. Install mysql server:
    * [ubuntu-16-04](https://www.digitalocean.com/community/tutorials/how-to-install-mysql-on-ubuntu-16-04)
    * [mac](https://dev.mysql.com/doc/refman/5.7/en/osx-installation-pkg.html)
    
2. Import the Database:
    * ubuntu and mac
    
        `mysql -u root infrastructure < infrastructure-database.sql`

_Note:_ If you set a password then change the config in database.py script 

#### To run the app: 
1. use the virtual environment "venv"
2. run app.py script
