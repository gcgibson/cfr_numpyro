
states="AS GU MP PR VI AL AK AZ AR CA CO CT DE DC FL GA HI ID IL IN IA KS KY LA ME MD MA MI MN MS MO MT NE NV NH NJ NM NY NC ND OH OK OR PA RI SC SD TN TX UT VT VA WA WV WI WY"

forecast_dates="2020-05-02 2020-05-09 2020-05-16 2020-05-23 2020-05-30 2020-06-06 2020-06-13 2020-06-20 2020-06-27 2020-07-04 2020-07-11 2020-07-18 2020-07-25 2020-08-01 2020-08-08 2020-08-15 2020-08-22 2020-08-29 2020-09-05 2020-09-12 2020-09-19 2020-09-26 2020-10-03 2020-10-10 2020-10-17 2020-10-24 2020-10-31 2020-11-07 2020-11-14 2020-11-21 2020-11-28 2020-12-05 2020-12-12 2020-12-19 2020-12-26"


for forecast_date in $forecast_dates; do
       echo "prefix is $prefix"

       for state in $states; do
            name=$state-$forecast_date

            logdir=log/$forecast_date
            [ -d $logdir ] || mkdir -p $logdir

            echo "launching $name"

            sbatch --job-name=$name \
                --output=$logdir/$state.out \
                --error=$logdir/$state.err \
                --nodes=1 \
                --ntasks=1 \
                --mem=4000 \
                --partition=defq \
                ./run_place.sh $state $forecast_date

            sleep 0.1
       done
done
