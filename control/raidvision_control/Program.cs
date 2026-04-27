using System;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text.Json;
using System.Text.Json.Serialization;
using System.Threading;
using WindowsDisplayAPI;

class Program
{
    private const double NeutralBrightness = 0.5;
    private const double NeutralContrast = 0.5;
    private const double NeutralGamma = 1.0;

    private static ControlProfile? _lastAppliedProfile = null;

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
                "apply-profile" => ApplyProfile(args),
                "custom-lut" => ApplyCustomLut(args),
                "watch" => Watch(args),
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

        return ApplyValues(displayIndex, brightness, contrast, gamma);
    }

    private static int ApplyCustomLut(string[] args)
    {
        int displayIndex = GetIntArg(args, "--display", 1);

        double blackPoint = GetDoubleArg(args, "--black-point", 0.02);
        double shadowLift = GetDoubleArg(args, "--shadow-lift", 0.35);
        double shadowPivot = GetDoubleArg(args, "--shadow-pivot", 0.35);
        double midtoneBoost = GetDoubleArg(args, "--midtone", 0.12);
        double midtoneWidth = GetDoubleArg(args, "--midtone-width", 0.28);
        double contrastStrength = GetDoubleArg(args, "--contrast", 0.12);
        double highlightProtect = GetDoubleArg(args, "--highlight-protect", 0.40);
        double highlightPivot = GetDoubleArg(args, "--highlight-pivot", 0.72);
        double greenBias = GetDoubleArg(args, "--green-bias", 0.025);
        double blueBias = GetDoubleArg(args, "--blue-bias", 0.035);

        blackPoint = Clamp(blackPoint, 0.0, 0.12);
        shadowLift = Clamp(shadowLift, 0.0, 1.0);
        shadowPivot = Clamp(shadowPivot, 0.18, 0.55);
        midtoneBoost = Clamp(midtoneBoost, 0.0, 0.50);
        midtoneWidth = Clamp(midtoneWidth, 0.12, 0.50);
        contrastStrength = Clamp(contrastStrength, 0.0, 0.50);
        highlightProtect = Clamp(highlightProtect, 0.0, 1.0);
        highlightPivot = Clamp(highlightPivot, 0.55, 0.90);
        greenBias = Clamp(greenBias, 0.0, 0.08);
        blueBias = Clamp(blueBias, 0.0, 0.08);

        Display target = GetDisplay(displayIndex);

        DisplayGammaRamp ramp = BuildVisibilityLut(
            blackPoint,
            shadowLift,
            shadowPivot,
            midtoneBoost,
            midtoneWidth,
            contrastStrength,
            highlightProtect,
            highlightPivot,
            greenBias,
            blueBias
        );

        target.GammaRamp = ramp;

        Console.WriteLine("Applied RaidVision custom LUT");
        Console.WriteLine($"Display: {displayIndex} | {target.DisplayName}");
        Console.WriteLine($"Black Point: {blackPoint:F3}");
        Console.WriteLine($"Shadow Lift: {shadowLift:F3}");
        Console.WriteLine($"Shadow Pivot: {shadowPivot:F3}");
        Console.WriteLine($"Midtone Boost: {midtoneBoost:F3}");
        Console.WriteLine($"Midtone Width: {midtoneWidth:F3}");
        Console.WriteLine($"Contrast: {contrastStrength:F3}");
        Console.WriteLine($"Highlight Protect: {highlightProtect:F3}");
        Console.WriteLine($"Highlight Pivot: {highlightPivot:F3}");
        Console.WriteLine($"Green Bias: {greenBias:F3}");
        Console.WriteLine($"Blue Bias: {blueBias:F3}");

        return 0;
    }


    private static DisplayGammaRamp BuildVisibilityLut(
        double blackPoint,
        double shadowLift,
        double shadowPivot,
        double midtoneBoost,
        double midtoneWidth,
        double contrastStrength,
        double highlightProtect,
        double highlightPivot,
        double greenBias,
        double blueBias
    )
    {
        ushort[] red = new ushort[256];
        ushort[] green = new ushort[256];
        ushort[] blue = new ushort[256];

        for (int i = 0; i < 256; i++)
        {
            double x = i / 255.0;

            double y = ApplyVisibilityCurve(
                x,
                blackPoint,
                shadowLift,
                shadowPivot,
                midtoneBoost,
                midtoneWidth,
                contrastStrength,
                highlightProtect,
                highlightPivot
            );

            double redY = y;
            double greenY = ApplyColorBias(y, x, greenBias, 0.0);
            double blueY = ApplyColorBias(y, x, 0.0, blueBias);

            red[i] = ToUShort(redY);
            green[i] = ToUShort(greenY);
            blue[i] = ToUShort(blueY);
        }

        return new DisplayGammaRamp(red, green, blue);
    }


    private static double ApplyVisibilityCurve(
        double x,
        double blackPoint,
        double shadowLift,
        double shadowPivot,
        double midtoneBoost,
        double midtoneWidth,
        double contrastStrength,
        double highlightProtect,
        double highlightPivot
    )
    {
        double y = x;

        if (blackPoint > 0.0)
        {
            y = Math.Max(0.0, (y - blackPoint) / (1.0 - blackPoint));
        }

        if (x < shadowPivot)
        {
            double t = Clamp(x / shadowPivot, 0.0, 1.0);
            double liftShape = 1.0 - Math.Pow(t, 1.85);
            double lifted = y + shadowLift * shadowPivot * liftShape;

            y = Math.Max(y, lifted);
        }

        if (contrastStrength > 0.0)
        {
            double contrastCenter = 0.42;
            double contrastWindow = 0.42;
            double distance = Math.Abs(y - contrastCenter) / contrastWindow;
            double weight = Clamp(1.0 - distance, 0.0, 1.0);

            y = contrastCenter + (y - contrastCenter) * (1.0 + contrastStrength * weight);
        }

        if (midtoneBoost > 0.0)
        {
            double centerDistance = Math.Abs(y - 0.50) / midtoneWidth;
            double midWeight = Clamp(1.0 - centerDistance, 0.0, 1.0);

            y += midtoneBoost * 0.12 * midWeight;
        }

        if (y > highlightPivot)
        {
            double t = Clamp((y - highlightPivot) / (1.0 - highlightPivot), 0.0, 1.0);
            double compression = highlightProtect * 0.18 * t * t;

            y -= compression;
        }

        y = Clamp(y, 0.0, 1.0);

        return y;
    }


    private static double ApplyColorBias(double y, double originalX, double greenBoost, double blueBoost)
    {
        if (originalX > 0.45)
        {
            return y;
        }

        double shadowWeight = Clamp((0.45 - originalX) / 0.45, 0.0, 1.0);
        double boost = (greenBoost + blueBoost) * shadowWeight;

        return Clamp(y + boost, 0.0, 1.0);
    }


    private static ushort ToUShort(double value)
    {
        value = Clamp(value, 0.0, 1.0);
        return (ushort)Math.Round(value * 65535.0);
    }

    private static int ApplyProfile(string[] args)
    {
        string path = GetStringArg(args, "--path", "control_profile.json");
        ControlProfile profile = ReadProfile(path);

        return ApplyProfileValues(profile, forceApply: true);
    }

    private static int Watch(string[] args)
    {
        string path = GetStringArg(args, "--path", "control_profile.json");
        string fullPath = Path.GetFullPath(path);
        string? directory = Path.GetDirectoryName(fullPath);
        string fileName = Path.GetFileName(fullPath);

        if (directory == null)
        {
            throw new InvalidOperationException("Invalid profile path.");
        }

        Console.WriteLine("RaidVision Control Watcher");
        Console.WriteLine($"Watching: {fullPath}");
        Console.WriteLine("Press Ctrl+C to stop and reset display.");

        bool keepRunning = true;

        Console.CancelKeyPress += (sender, eventArgs) =>
        {
            eventArgs.Cancel = true;
            keepRunning = false;

            Console.WriteLine();
            Console.WriteLine("Stopping watcher...");
        };

        if (File.Exists(fullPath))
        {
            TryApplyProfile(fullPath, forceApply: true);
        }
        else
        {
            Console.WriteLine("Profile file does not exist yet. Waiting for updates...");
        }

        using FileSystemWatcher watcher = new FileSystemWatcher(directory, fileName);

        watcher.NotifyFilter =
            NotifyFilters.LastWrite |
            NotifyFilters.FileName |
            NotifyFilters.Size |
            NotifyFilters.CreationTime;

        watcher.Changed += (_, _) => TryApplyProfile(fullPath, forceApply: false);
        watcher.Created += (_, _) => TryApplyProfile(fullPath, forceApply: false);
        watcher.Renamed += (_, _) => TryApplyProfile(fullPath, forceApply: false);

        watcher.EnableRaisingEvents = true;

        while (keepRunning)
        {
            Thread.Sleep(250);
        }

        ResetAllKnownDisplays();

        Console.WriteLine("Watcher stopped.");
        return 0;
    }

    private static void TryApplyProfile(string path, bool forceApply)
    {
        try
        {
            Thread.Sleep(100);

            ControlProfile profile = ReadProfile(path);
            ApplyProfileValues(profile, forceApply);
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Profile apply skipped: {ex.Message}");
        }
    }

    private static ControlProfile ReadProfile(string path)
    {
        if (!File.Exists(path))
        {
            throw new FileNotFoundException($"Profile file not found: {path}");
        }

        string json = File.ReadAllText(path);

        JsonSerializerOptions options = new JsonSerializerOptions
        {
            PropertyNameCaseInsensitive = true
        };

        ControlProfile? profile = JsonSerializer.Deserialize<ControlProfile>(json, options);

        if (profile == null)
        {
            throw new InvalidOperationException("Could not parse control profile.");
        }

        return profile;
    }

    private static int ApplyProfileValues(ControlProfile profile, bool forceApply)
    {
        if (profile.Reset)
        {
            return ResetDisplay(profile.DisplayIndex);
        }

        if (!forceApply && _lastAppliedProfile != null && !IsMeaningfulChange(profile, _lastAppliedProfile))
        {
            Console.WriteLine("Profile update ignored. Change below threshold.");
            return 0;
        }

        int result = ApplyValues(
            profile.DisplayIndex,
            profile.Brightness,
            profile.Contrast,
            profile.Gamma
        );

        _lastAppliedProfile = profile;
        return result;
    }

    private static bool IsMeaningfulChange(ControlProfile current, ControlProfile previous)
    {
        if (current.DisplayIndex != previous.DisplayIndex)
        {
            return true;
        }

        if (current.Reset != previous.Reset)
        {
            return true;
        }

        double brightnessDelta = Math.Abs(current.Brightness - previous.Brightness);
        double contrastDelta = Math.Abs(current.Contrast - previous.Contrast);
        double gammaDelta = Math.Abs(current.Gamma - previous.Gamma);

        return brightnessDelta >= 0.02 ||
               contrastDelta >= 0.02 ||
               gammaDelta >= 0.02;
    }

    private static int ApplyValues(int displayIndex, double brightness, double contrast, double gamma)
    {
        brightness = Clamp(brightness, 0.0, 1.0);
        contrast = Clamp(contrast, 0.0, 1.0);
        gamma = Clamp(gamma, 0.4, 2.8);

        Display target = GetDisplay(displayIndex);

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
        return ResetDisplay(displayIndex);
    }

    private static int ResetDisplay(int displayIndex)
    {
        Display target = GetDisplay(displayIndex);

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

    private static void ResetAllKnownDisplays()
    {
        try
        {
            var displays = Display.GetDisplays().ToList();

            for (int i = 0; i < displays.Count; i++)
            {
                displays[i].GammaRamp = new DisplayGammaRamp(
                    NeutralBrightness,
                    NeutralContrast,
                    NeutralGamma
                );

                Console.WriteLine($"Reset display {i}: {displays[i].DisplayName}");
            }
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Reset all failed: {ex.Message}");
        }
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

    private static string GetStringArg(string[] args, string name, string defaultValue)
    {
        int index = Array.IndexOf(args, name);

        if (index < 0 || index + 1 >= args.Length)
        {
            return defaultValue;
        }

        return args[index + 1];
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
        Console.WriteLine("  apply --display 1 --brightness 0.60 --contrast 0.55 --gamma 1.35");
        Console.WriteLine("  apply-profile --path ..\\..\\control_profile.json");
        Console.WriteLine("  custom-lut --display 1 --black-point 0.02 --shadow-lift 0.45 --shadow-pivot 0.35 --midtone 0.12 --midtone-width 0.28 --contrast 0.12 --highlight-protect 0.45 --highlight-pivot 0.72 --green-bias 0.025 --blue-bias 0.035");
        Console.WriteLine("  watch --path ..\\..\\control_profile.json");
        Console.WriteLine("  reset --display 1");
        Console.WriteLine();
        Console.WriteLine("Neutral values:");
        Console.WriteLine("  brightness = 0.5");
        Console.WriteLine("  contrast   = 0.5");
        Console.WriteLine("  gamma      = 1.0");
    }
}

public class ControlProfile
{
    [JsonPropertyName("display_index")]
    public int DisplayIndex { get; set; } = 1;

    [JsonPropertyName("brightness")]
    public double Brightness { get; set; } = 0.5;

    [JsonPropertyName("contrast")]
    public double Contrast { get; set; } = 0.5;

    [JsonPropertyName("gamma")]
    public double Gamma { get; set; } = 1.0;

    [JsonPropertyName("reset")]
    public bool Reset { get; set; } = false;

    [JsonPropertyName("heartbeat")]
    public string? Heartbeat { get; set; }
}