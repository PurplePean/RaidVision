using System;
using System.Globalization;
using System.Linq;
using WindowsDisplayAPI;

class Program
{
    private const double NeutralBrightness = 0.5;
    private const double NeutralContrast = 0.5;
    private const double NeutralGamma = 1.0;

    static int Main(string[] args)
    {
        if (args.Length == 0)
        {
            PrintHelp();
            return 1;
        }

        try
        {
            string command = args[0].ToLowerInvariant();

            return command switch
            {
                "list" => ListDisplays(),
                "apply" => Apply(args),
                "reset" => Reset(args),
                _ => UnknownCommand(command)
            };
        }
        catch (Exception ex)
        {
            Console.WriteLine($"ERROR: {ex.Message}");
            return 1;
        }
    }

    private static int ListDisplays()
    {
        var displays = Display.GetDisplays().ToList();

        Console.WriteLine("RaidVision Displays");

        for (int i = 0; i < displays.Count; i++)
        {
            Console.WriteLine($"{i}: {displays[i].DisplayName}");
        }

        return 0;
    }

    private static int Apply(string[] args)
    {
        int displayIndex = GetIntArg(args, "--display", 0);

        double brightness = GetDoubleArg(args, "--brightness", NeutralBrightness);
        double contrast = GetDoubleArg(args, "--contrast", NeutralContrast);
        double gamma = GetDoubleArg(args, "--gamma", NeutralGamma);

        brightness = Clamp(brightness, 0.0, 1.0);
        contrast = Clamp(contrast, 0.0, 1.0);
        gamma = Clamp(gamma, 0.4, 2.8);

        var target = GetDisplay(displayIndex);

        target.GammaRamp = new DisplayGammaRamp(brightness, contrast, gamma);

        Console.WriteLine("Applied RaidVision display profile");
        Console.WriteLine($"Display: {displayIndex} | {target.DisplayName}");
        Console.WriteLine($"Brightness: {brightness:F3}");
        Console.WriteLine($"Contrast: {contrast:F3}");
        Console.WriteLine($"Gamma: {gamma:F3}");

        return 0;
    }

    private static int Reset(string[] args)
    {
        int displayIndex = GetIntArg(args, "--display", 0);

        var target = GetDisplay(displayIndex);

        target.GammaRamp = new DisplayGammaRamp(
            NeutralBrightness,
            NeutralContrast,
            NeutralGamma
        );

        Console.WriteLine("Reset RaidVision display profile");
        Console.WriteLine($"Display: {displayIndex} | {target.DisplayName}");
        Console.WriteLine($"Brightness: {NeutralBrightness:F3}");
        Console.WriteLine($"Contrast: {NeutralContrast:F3}");
        Console.WriteLine($"Gamma: {NeutralGamma:F3}");

        return 0;
    }

    private static Display GetDisplay(int displayIndex)
    {
        var displays = Display.GetDisplays().ToList();

        if (displayIndex < 0 || displayIndex >= displays.Count)
        {
            throw new ArgumentOutOfRangeException(
                nameof(displayIndex),
                $"Display index {displayIndex} is invalid. Available range: 0 to {displays.Count - 1}"
            );
        }

        return displays[displayIndex];
    }

    private static int GetIntArg(string[] args, string name, int defaultValue)
    {
        int index = Array.IndexOf(args, name);

        if (index < 0 || index + 1 >= args.Length)
        {
            return defaultValue;
        }

        return int.Parse(args[index + 1], CultureInfo.InvariantCulture);
    }

    private static double GetDoubleArg(string[] args, string name, double defaultValue)
    {
        int index = Array.IndexOf(args, name);

        if (index < 0 || index + 1 >= args.Length)
        {
            return defaultValue;
        }

        return double.Parse(args[index + 1], CultureInfo.InvariantCulture);
    }

    private static double Clamp(double value, double min, double max)
    {
        return Math.Max(min, Math.Min(max, value));
    }

    private static int UnknownCommand(string command)
    {
        Console.WriteLine($"Unknown command: {command}");
        PrintHelp();
        return 1;
    }

    private static void PrintHelp()
    {
        Console.WriteLine("RaidVision Control CLI");
        Console.WriteLine();
        Console.WriteLine("Commands:");
        Console.WriteLine("  list");
        Console.WriteLine("  apply --display 0 --brightness 0.60 --contrast 0.55 --gamma 1.35");
        Console.WriteLine("  reset --display 0");
        Console.WriteLine();
        Console.WriteLine("Neutral values:");
        Console.WriteLine("  brightness = 0.5");
        Console.WriteLine("  contrast   = 0.5");
        Console.WriteLine("  gamma      = 1.0");
    }
}