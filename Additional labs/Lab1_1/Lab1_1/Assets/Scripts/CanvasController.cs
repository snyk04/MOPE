using System;
using System.Globalization;
using UnityEngine;
using UnityEngine.UI;

public class CanvasController : MonoBehaviour
{
    [SerializeField] private MessageBox messageBox;

    [SerializeField] private InputField inputN;
    [SerializeField] private Text result;

    private void Awake()
    {
        CultureInfo.CurrentCulture = CultureInfo.GetCultureInfo("en-US");
    }
    private void Start()
    {
        messageBox.SetTitle("Помилка");
    }

    private bool CheckInputs(out long n)
    {
        if (long.TryParse(inputN.text, NumberStyles.Float, CultureInfo.InvariantCulture, out n))
        {
            return true;
        }
        n = 0;
        return false;
    }
    
    public void OnFindMultipliers()
    {
        if (!CheckInputs(out var n))
        {
            messageBox.SetMessage("Число n введено некоректно!");
            messageBox.Show();
            return;
        }

        long[] multipliers;
        int amountOfOperations;
        try
        {
            multipliers = FindMultipliers(n, out var numOfOperations);
            amountOfOperations = numOfOperations;
        }
        catch (Exception e)
        {
            messageBox.SetMessage(e.Message);
            messageBox.Show();
            return;
        }

        ShowResult(multipliers, n, amountOfOperations);
    }

    private void ShowResult(long[] multipliers, long n, int amountOfOperations)
    {
        var resultText = $"n = {n} = {string.Join(" * ", multipliers)}\n" +
                         $"Кількість проведених ітерацій: {amountOfOperations}";
        result.text = resultText;
    }

    private long[] FindMultipliers(long n, out int numOfOperations)
    {
        var result = FermatMethod.Factorize(n, out var amountOfOperations);
        numOfOperations = amountOfOperations;
        return result;
    }
}