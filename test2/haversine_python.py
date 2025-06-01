from numpy import sin, cos, arccos, pi, round

class haversine():
    def rad2deg(self , radians):
        degrees = radians * 180 / pi
        return degrees

    def deg2rad(self , degrees):
        radians = degrees * pi / 180
        return radians

    def getDistanceBetweenPointsNew(self , latitude1, longitude1, latitude2, longitude2, unit = 'kilometers'):
        
        theta = longitude1 - longitude2
        
        distance = 60 * 1.1515 * self.rad2deg(
            arccos(
                (sin(self.deg2rad(latitude1)) * sin(self.deg2rad(latitude2))) + 
                (cos(self.deg2rad(latitude1)) * cos(self.deg2rad(latitude2)) * cos(self.deg2rad(theta)))
            )
        )
        
        if unit == 'miles':
            return round(distance, 2)
        if unit == 'kilometers':
            return round(distance * 1.609344, 2)