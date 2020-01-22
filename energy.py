

class Potential():

    evaluate_count = 0

    def evaluate(self, X_dict:dict, solution:list, initial_key:int, speed=1.0):
        self.evaluate_count += 1

        def in_window(t, ti, tf):
            return (t >= ti) and (t <= tf)

        def distance_to_window(t, ti, tf):
            if t < ti:
                return ti - t
            else:
                return t - tf

        cost, t = 0, 0
        prev_key = initial_key
        
        # Check that every point is taken once
        every_point_taken = all([key==initial_key or key in solution for key in X_dict])
        try:
            assert(every_point_taken)
        except AssertionError:
            print('Every point must be taken once !')
            return (-1, -1)

        for key in solution:

            # Pass the first key if it is the initial key
            if key == prev_key:
                continue

            # Compute time taken
            distance = ( (X_dict[key]['x'] - X_dict[prev_key]['x'])**2 \
                        + (X_dict[key]['y'] - X_dict[prev_key]['y'])**2 )**.5
            
            time_taken = distance/speed
            t += time_taken
            
            # Compute cost if not in window
            ti, tf = X_dict[key]['ti'], X_dict[key]['tf']
            print('\n', key, (t, ti, tf), (X_dict[key]['x'], X_dict[key]['y']), (X_dict[prev_key]['x'], X_dict[prev_key]['y']))
            if not in_window(t, ti, tf):
                print('Not in window !', distance_to_window(t, ti, tf))
                cost += distance_to_window(t, ti, tf)

            # Update the previous key
            prev_key = key    
            
        return cost, t


