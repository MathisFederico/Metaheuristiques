from utilities import isValid

class Potential():

    evaluate_count = 0
    dist_count = 0
    speed = 1.0

    def get_dist(self, X_dict:dict, key1:int, key2:int):
        self.dist_count += 1
        distance = ( (X_dict[key1]['x'] - X_dict[key2]['x'])**2 \
                    + (X_dict[key1]['y'] - X_dict[key2]['y'])**2 )**.5
        return distance

    def evaluate(self, X_dict:dict, solution:list, initial_key:int):
        self.evaluate_count += 1

        def in_window(t, ti, tf):
            return (t >= ti) and (t <= tf)

        def distance_to_window(t, ti, tf):
            if t < ti:
                return ti - t
            else:
                return t - tf

        cost, t, errors = 0, 0, 0
        prev_key = initial_key
        
        # Check that every point is taken once
        if not isValid(X_dict, solution, initial_key):
            return (-1, -1)

        for key in solution:

            # Pass the first key if it is the initial key
            if key == prev_key:
                continue

            # Compute time taken
            distance = self.get_dist(X_dict, key, prev_key)
            time_taken = distance/self.speed
            t += time_taken
            
            # Compute cost if not in window
            ti, tf = X_dict[key]['ti'], X_dict[key]['tf']
            # print('\n', key, (t, ti, tf), (X_dict[key]['x'], X_dict[key]['y']), (X_dict[prev_key]['x'], X_dict[prev_key]['y']))
            if not in_window(t, ti, tf):
                # print('Not in window !', distance_to_window(t, ti, tf))
                errors +=1
                cost += distance_to_window(t, ti, tf)

            # Update the previous key
            prev_key = key    
            
        return cost, t, errors


