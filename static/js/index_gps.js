let map;
let carIcon = {
    url: "static/images/car.png",
    scaledSize: new google.maps.Size(30, 30)
};

let selectedCarIcon = {
    url: "static/images/car.png",
    scaledSize: new google.maps.Size(40, 40)
};

let stationIcon = {
    url: "static/images/station.png",
    scaledSize: new google.maps.Size(30, 30)
};

let selectedStationIcon = {
    url: "static/images/station.png",
    scaledSize: new google.maps.Size(40, 40)
};

let carMarkers=[];
let stationMarkers=[];
let gpsCoords = [];
let tuberlin = new google.maps.LatLng(52.512230, 13.327135);
let directionsService;
let directionsDisplay;
let timer; 

$(document).ready(function() {
    $("#cars").html("");
    $("#stations").html("");
    loadCars(carList);
    loadGrids(gridList);
    initMap();
});

function initMap(){
    map = new google.maps.Map(document.getElementById('map'), {
        zoom: 13,
        center: new google.maps.LatLng(52.5159995,13.3560001),
        mapTypeId: google.maps.MapTypeId.ROADMAP
    });
    // init directionsService 
    directionsService = new google.maps.DirectionsService;
    directionsDisplay = new google.maps.DirectionsRenderer;
    directionsDisplay.setMap(map);
}

function loadCars(Auto_cars){
    for (let i = 0 ; i < Auto_cars.length ; i++)
    {
        $("#cars").append( '<option value='+Auto_cars[i].id+'>car' + Auto_cars[i].id + '</option>' );
        carMarkers.push(addMarker('car'+ Auto_cars[i].id +'\nsoc: '+Auto_cars[i].soc
            , new google.maps.LatLng(Auto_cars[i].lat,Auto_cars[i].lon),carIcon));
        // display routes
        calculateDisplayRoutes(Auto_cars[i].lat, Auto_cars[i].lon);
    }
    $("#cars").selectpicker("refresh");

    return Auto_cars;
}

// This function displays the route in driving mode between origin and fixed goal tu berlin
// https://developers.google.com/maps/documentation/javascript/examples/directions-simple
function calculateDisplayRoutes(lat, long) {
     let origin = new google.maps.LatLng(lat, long);
     let routePoints = [];
     let pointDuration = [];
     directionsService.route({
          origin: origin,
          destination: tuberlin,
          travelMode: 'DRIVING'
        }, function(response, status) {
          if (status === 'OK') {
            // comment following line to not display route            
            directionsDisplay.setDirections(response);
 		    for(var i=0; i<response.routes[0].legs[0].steps.length; i++) {
                // get all turning points of route
                routePoints.push(response.routes[0].legs[0].steps[i].end_location);
        // time in minutes for each part of route
                pointDuration.push(response.routes[0].legs[0].steps[i].duration.value/60);
            }
            calculateNewCarPosition(routePoints.unshift(origin), pointDuration);
          } else {
            window.alert('Directions request failed due to ' + status);
          }
        });
}

function calculateNewCarPosition(routePoints, pointDuration) {
    for (let i=0; i<routePoints.length-1; i++) {
        let start, end, distanceLat, distanceLong;
        start = routePoints[i];
        end = routePoints[i+1];
        distanceLat = start[0]-end[0];
        distanceLong = start[1]-end[1];
        increaseLat = distanceLat/Math.floor(pointDuration[i]);
        increaseLong = distanceLong/Math.floor(pointDuration[i]);
        for (let j=0; j<Math.floor(pointDuration[i]); j++) {
            // increase start coordinate towards end coordinate
            gpsCoords.push({routePoints[i][0]+increaseLat, routePoints[i][1]+increaseLong});
        }
        if (Math.floor(pointDuration[i]) < pointDuration[i]) {
            // perform last step of routePart
            gpsCoords.push(end);
        }
    }
    for (let i=0; i<gpsCoords.length; i++) {
        timer = setInterval(updateGPS, 1000, gpsCoords[i]);   
    }
}

function updateGPS() {
            // update car icon to gpsCoords i or do a smooth transition animation ? 
            // make coords available to optimization
            newcoords = gpsCoords[i];
            newIconCoords = new google.maps.LatLng(gpsCoords[i][0], gpsCoords[i][1]);
}

function loadGrids(gridList){
    for (var i = 0 ;i<gridList.length;i++)
    {
        $("#stations").append( '<option  value='+gridList[i].id+'>station' + gridList[i].id + '</option>' );
        stationMarkers.push(addMarker('station'+ gridList[i].id +'\nprice: '+gridList[i].price
            +'\ncapacity: '+gridList[i].capacity, new google.maps.LatLng(gridList[i].lat,gridList[i].lon),stationIcon));
    }
    $("#stations").selectpicker("refresh");

    return gridList;
}


$("#cars").change(function () {
    let selectedCar= $(this).find("option:selected").val();
        showCar(selectedCar);
});

$("#stations").change(function () {
    let selectedGrid= $(this).find("option:selected").val();
        showGrid(selectedGrid);
});

$("#searchButton").click(function(){
   //TODO
});

function addMarker(title,position,icon) {
    return new google.maps.Marker({
        title:title,
        position: position,
        map: map,
        icon: icon
    });
}




function showCar(selectedCar){
    for (let i =0 ; i < carList.length ; i++)
    {
        if (carList[i].id == selectedCar)
        {
            $("#cars").selectpicker("refresh");
            $('#location').val("(" + carList[i].lat + "," + carList[i].lon + ")");
            $('#speedInput').val(carList[i].speed);
            $('#battery').attr('src', getBatteryIcon(carList[i].soc * 100));
            $('#b1soc').text(carList[i].soc * 100 + '%');
            if (carList[i].soc <= 0.2)
            {
                $('#mode').removeAttr('disabled');
                $('#searchButton').removeAttr('hidden');
            }
            carMarkers.forEach(function (marker) {
                let carId = marker.title.split("\n")[0];
                if (carId == 'car'+carList[i].id)
                {
                    marker.setIcon(selectedCarIcon);
                    map.setCenter(marker.position);
                }
            });
        }
    }
}

function showGrid(selectedGrid){
    for (var i =0 ; i < gridList.length; i++){
        if (gridList[i].id == selectedGrid)
        {
            $("#stations").selectpicker("refresh");
            $('#stationLocation').val( "("+gridList[i].lat + ","+gridList[i].lon+")") ;
            $('#capacity').val(gridList[i].capacity);
            $('#price').val(gridList[i].price);
            stationMarkers.forEach(function (marker)
            {
                stationId = marker.title.split("\n")[0];
                if (stationId == 'station'+gridList[i].id)
                {
                    marker.setIcon(selectedStationIcon);
                    map.setCenter(marker.position);
                }
            });
        }
    }
}

function getBatteryIcon(soc){
    let src;
    if ( soc <= 20 && soc > 0 )
        src = 'static/images/20.png' ;
    else if ( soc <= 40 && soc > 20 )
        src = 'static/images/40.png' ;
    else if (soc < 60 && soc > 40  )
        src = 'static/images/60.png' ;
    else if (soc <= 80 && soc >= 60 )
        src = 'static/images/80.png' ;
    else if ( soc <= 100 && soc > 80  )
        src = 'static/images/100.png' ;
    return src ;
}
