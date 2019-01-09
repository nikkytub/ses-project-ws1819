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


$(document).ready(function() {
    initMap();
    $("#cars").html("");
    $("#stations").html("");
    loadCars(carList);
    loadGrids(gridList);
});

function initMap(){
    map = new google.maps.Map(document.getElementById('map'), {
        zoom: 13,
        center: new google.maps.LatLng(52.5159995,13.3560001),
        mapTypeId: google.maps.MapTypeId.ROADMAP
    });
}

function loadCars(Auto_cars){
    for (let i = 0 ; i < Auto_cars.length ; i++)
    {
        $("#cars").append( '<option value='+Auto_cars[i].id+'>car' + Auto_cars[i].id + '</option>' );
        carMarkers.push(addMarker('car'+ Auto_cars[i].id +'\nsoc: '+Auto_cars[i].soc
            , new google.maps.LatLng(Auto_cars[i].lat,Auto_cars[i].lon),carIcon));
    }
    $("#cars").selectpicker("refresh");

    return Auto_cars;
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
