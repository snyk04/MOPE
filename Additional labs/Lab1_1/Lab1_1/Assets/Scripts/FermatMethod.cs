using System;
using System.Collections.Generic;
using UnityEngine;

public static class FermatMethod
{
    public static long[] Factorize(long n)
    {
        if (n % 2 == 0)
        {
            throw new Exception("n повинне бути непарним!");
        }

        if (n <= 1)
        {
            throw new Exception("n повинно бути більше одиниці!");
        }

        var multipliers = new List<long>();
        var sqrts = GetSumOfSquares(n, out var numOfOperations);
        Debug.Log(numOfOperations);
        multipliers.Add(Math.Abs(sqrts[0] + sqrts[1]));
        multipliers.Add(Math.Abs(sqrts[0] - sqrts[1]));

        return multipliers.ToArray();
    }

    private static long[] GetSumOfSquares(long n, out int numOfOperations)
    {
        numOfOperations = 0;

        var x = Math.Ceiling(Math.Sqrt(n));
        var y = Math.Pow(x, 2) - n;

        while (Math.Abs(Math.Sqrt(y) - Math.Ceiling(Math.Sqrt(y))) > 0.0001f)
        {
            numOfOperations++;
            x++;
            y = Math.Pow(x, 2) - n;


            if (Time.deltaTime >= 30)
            {
                throw new Exception("Операція зайняла надто багато часу!");
            };
        }

        return new[] {(long) x, (long) Math.Sqrt(y)};
    }
}