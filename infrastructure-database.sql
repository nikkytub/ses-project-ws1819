-- MySQL dump 10.13  Distrib 5.7.25, for Linux (x86_64)
--
-- Host: localhost    Database: infrastructure
-- ------------------------------------------------------
-- Server version	5.7.25-0ubuntu0.16.04.2

/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!40101 SET NAMES utf8 */;
/*!40103 SET @OLD_TIME_ZONE=@@TIME_ZONE */;
/*!40103 SET TIME_ZONE='+00:00' */;
/*!40014 SET @OLD_UNIQUE_CHECKS=@@UNIQUE_CHECKS, UNIQUE_CHECKS=0 */;
/*!40014 SET @OLD_FOREIGN_KEY_CHECKS=@@FOREIGN_KEY_CHECKS, FOREIGN_KEY_CHECKS=0 */;
/*!40101 SET @OLD_SQL_MODE=@@SQL_MODE, SQL_MODE='NO_AUTO_VALUE_ON_ZERO' */;
/*!40111 SET @OLD_SQL_NOTES=@@SQL_NOTES, SQL_NOTES=0 */;

--
-- Table structure for table `car`
--

DROP TABLE IF EXISTS `car`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `car` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `speed` tinyint(3) unsigned DEFAULT NULL,
  `capacity` tinyint(3) unsigned DEFAULT NULL,
  `availability` tinyint(1) DEFAULT NULL,
  `lat` double DEFAULT NULL,
  `lon` double DEFAULT NULL,
  `soc` double DEFAULT NULL,
  `powerPD` double DEFAULT NULL,
  `mode` text,
  `powerKm` double DEFAULT NULL,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB AUTO_INCREMENT=3 DEFAULT CHARSET=latin1;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `car`
--

LOCK TABLES `car` WRITE;
/*!40000 ALTER TABLE `car` DISABLE KEYS */;
INSERT INTO `car` VALUES (1,50,75,1,52.5601315,13.4177424,0.21,15,'eco_mode',0.15),(2,50,75,0,52.5054011,13.43636,0.21,15,'costSaving_mode',0.15);
/*!40000 ALTER TABLE `car` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `grid`
--

DROP TABLE IF EXISTS `grid`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `grid` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `capacity` tinyint(3) unsigned DEFAULT NULL,
  `availability` tinyint(1) DEFAULT NULL,
  `lat` double DEFAULT NULL,
  `lon` double DEFAULT NULL,
  `price` double DEFAULT NULL,
  `total_charge_needed_at_grid` double DEFAULT '0',
  `name` char(20) DEFAULT NULL,
  `alpha` double DEFAULT NULL,
  `p_charging_station` double DEFAULT NULL,
  `dist` double DEFAULT '0',
  `alpha_next` double DEFAULT NULL,
  `price_next` double DEFAULT NULL,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB AUTO_INCREMENT=5 DEFAULT CHARSET=latin1;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `grid`
--

LOCK TABLES `grid` WRITE;
/*!40000 ALTER TABLE `grid` DISABLE KEYS */;
INSERT INTO `grid` VALUES (1,80,1,52.5247661,13.4027603,1.4,0,'grid1',0.7,0.5,0,NULL,NULL),(2,100,1,52.5191382,13.3893115,1.2,0,'grid2',0.5,0.5,0,NULL,NULL),(3,130,1,52.5065186,13.3193903,0.8,0,'grid3',0.3,1,0,NULL,NULL),(4,100,1,52.5164063,13.3478676,0.33,0,'grid4',0.4,1,0,NULL,NULL);
/*!40000 ALTER TABLE `grid` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `grid2`
--

DROP TABLE IF EXISTS `grid2`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `grid2` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `availability` tinyint(4) DEFAULT NULL,
  `lat` double DEFAULT NULL,
  `lon` double DEFAULT NULL,
  `price` double DEFAULT NULL,
  `total_charge_needed_at_grid` double DEFAULT '0',
  `name` char(20) DEFAULT NULL,
  `alpha` double DEFAULT NULL,
  `p_charging_station` double,
  `dist` double DEFAULT '0',
  `alpha_next` double DEFAULT NULL,
  `price_next` double DEFAULT NULL,
  PRIMARY KEY (`id`),
  UNIQUE KEY `grid2_name_uindex` (`name`)
) ENGINE=InnoDB AUTO_INCREMENT=2612 DEFAULT CHARSET=latin1;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `grid2`
--

LOCK TABLES `grid2` WRITE;
/*!40000 ALTER TABLE `grid2` DISABLE KEYS */;
/*!40000 ALTER TABLE `grid2` ENABLE KEYS */;
UNLOCK TABLES;
/*!40103 SET TIME_ZONE=@OLD_TIME_ZONE */;

/*!40101 SET SQL_MODE=@OLD_SQL_MODE */;
/*!40014 SET FOREIGN_KEY_CHECKS=@OLD_FOREIGN_KEY_CHECKS */;
/*!40014 SET UNIQUE_CHECKS=@OLD_UNIQUE_CHECKS */;
/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
/*!40111 SET SQL_NOTES=@OLD_SQL_NOTES */;

-- Dump completed on 2019-02-04 21:00:03
