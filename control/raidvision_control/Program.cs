using System;
using System.Linq;
using WindowsDisplayAPI;

class Program
{
    static void Main()
    {
        Console.WriteLine("RaidVision Control Test");

        var displays = Display.GetDisplays().ToList();

        for (int i = 0; i < displays.Count; i++)
        {
            Console.WriteLine($"{i}: {displays[i].DisplayName}");
        }

        Console.WriteLine();
        Console.Write("Pick display index to test: ");
        int index = int.Parse(Console.ReadLine() ?? "0");

        var target = displays[index];

        Console.WriteLine("Applying test gamma ramp...");
        target.GammaRamp = new DisplayGammaRamp(
            brightness: 0.60,
            contrast: 0.55,
            gamma: 1.20
        );

        Console.WriteLine("Applied. Press Enter to reset.");
        Console.ReadLine();

        target.GammaRamp = new DisplayGammaRamp(0.5, 0.5, 1.0);

        Console.WriteLine("Reset complete.");
    }
}